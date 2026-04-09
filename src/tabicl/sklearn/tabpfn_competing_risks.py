from __future__ import annotations

"""
TabPFNCompetingRisks — Competing risks survival model with frozen TabPFN embeddings.

Architecture:
  1. TabPFN (frozen pretrained transformer) acts as a feature encoder: X → embedding.
  2. K separate _MLPRisk Cox heads (one per competing event type) are trained on the
     frozen embeddings using the Breslow partial log-likelihood.

Key difference from TabICL:
  TabPFN is an in-context learner — its representations for a query sample depend on
  the training context passed to it.  To avoid leakage when embedding training samples,
  we use out-of-fold (OOF) extraction:  each fold of the training data is embedded using
  the remaining folds as context.  Validation / test samples are embedded using the
  full training set as context (standard inference).

Dummy labels for TabPFN context:
  Because TabPFN requires class labels for its context, we construct a binary label
  y_dummy = (event_type > 0), i.e. "any event occurred" vs "censored".  This is the
  most natural survival-aware label that can be derived without time information.

Author: Yarin
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from packaging import version
import sklearn

if TYPE_CHECKING:
    import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")

from .survivor import (
    _to_tensor,
    _coxph_breslow_loss,
    _breslow_baseline_hazard,
    _concordance_index,
    _MLPRisk,
)


class TabPFNCompetingRisks(BaseEstimator):
    """
    Competing risks survival model using frozen TabPFN embeddings + cause-specific Cox heads.

    Parameters
    ----------
    n_event_types : int
        Number of competing event types K.
        Event type 0 = censored; types 1 … K are the competing events.

    n_oof_splits : int, default=5
        Number of folds used for out-of-fold embedding of *training* samples.
        Higher values reduce leakage but increase fitting time.

    max_context_samples : int or None, default=None
        If set, subsample the TabPFN context to at most this many samples.
        Useful when the training set exceeds TabPFN's recommended limit
        (~10 000 samples for v2.6).  Set to None to use all training data
        (requires ignore_pretraining_limits=True).

    ignore_pretraining_limits : bool, default=True
        Pass ignore_pretraining_limits=True to TabPFNClassifier so that
        datasets exceeding TabPFN's default size / feature limits are accepted.

    hidden : int or None, default=None
        Hidden layer size for each Cox risk head.
        None → linear model (single Linear layer).

    dropout : float, default=0.0
        Dropout rate inside each Cox head (only effective when hidden is set).

    lr : float, default=1e-3
        Learning rate for AdamW.

    weight_decay : float, default=1e-4
        Weight decay for AdamW.

    epochs : int, default=200
        Maximum training epochs per cause-specific head.

    patience : int, default=20
        Early-stopping patience (epochs without validation improvement).

    device : str or torch.device or None, default=None
        Device for the Cox heads.  None → auto-detect (CUDA if available, else CPU).
        TabPFN uses its own internal device setting via the same string.

    random_state : int, default=42
        Random seed for OOF splits, weight initialisation, and TabPFN.

    verbose : bool, default=False
        Print progress during fitting.

    Attributes
    ----------
    tabpfn_model_ : TabPFNClassifier
        The TabPFN model fitted on the full training set (used for val/test embedding).

    models_ : list of _MLPRisk
        K fitted cause-specific Cox heads.

    baseline_cum_hazards_ : list of np.ndarray
        Breslow baseline cumulative hazards, one per cause.

    event_times_per_cause_ : list of np.ndarray
        Event times associated with each baseline hazard.

    event_times_ : np.ndarray
        Union of all per-cause event times (sorted).

    device_ : torch.device

    Examples
    --------
    >>> import numpy as np
    >>> from tabicl.sklearn.tabpfn_competing_risks import TabPFNCompetingRisks
    >>>
    >>> X = np.random.randn(500, 10)
    >>> times = np.random.exponential(100, 500)
    >>> events = np.random.choice([0, 1, 2], size=500, p=[0.3, 0.4, 0.3])
    >>>
    >>> model = TabPFNCompetingRisks(n_event_types=2, hidden=64, epochs=100, verbose=True)
    >>> model.fit(X[:400], (times[:400], events[:400]),
    ...           X_val=X[400:], y_val=(times[400:], events[400:]))
    >>> risks = model.predict_cause_specific_risk(X[400:])   # (100, 2)
    >>> c = model.score(X[400:], (times[400:], events[400:]))
    """

    def __init__(
        self,
        n_event_types: int,
        # TabPFN / embedding params
        n_oof_splits: int = 5,
        max_context_samples: Optional[int] = None,
        ignore_pretraining_limits: bool = True,
        # Cox head params
        hidden: Optional[int] = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        patience: int = 20,
        # General
        device: Optional[Union[str, torch.device]] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_event_types = n_event_types
        self.n_oof_splits = n_oof_splits
        self.max_context_samples = max_context_samples
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _more_tags(self):
        return dict(non_deterministic=True)

    # ------------------------------------------------------------------
    # TabPFN helpers
    # ------------------------------------------------------------------

    def _tabpfn_device_str(self) -> str:
        """Convert torch.device to the string format TabPFN expects."""
        d = str(self.device_)
        # TabPFN v2 accepts 'cpu', 'cuda', or 'mps' — not 'cuda:0'
        if d.startswith("cuda"):
            return "cuda"
        return d  # 'cpu' or 'mps'

    def _make_tabpfn(self):
        """
        Instantiate a fresh TabPFNClassifier.

        TabPFN's weights are *always* frozen (pretrained foundation model).
        No gradient updates are performed on TabPFN during our training.
        """
        from tabpfn import TabPFNClassifier

        kwargs = dict(
            device=self._tabpfn_device_str(),
            random_state=self.random_state,
        )
        # ignore_pretraining_limits is available in TabPFN v2+
        try:
            import inspect
            sig = inspect.signature(TabPFNClassifier.__init__)
            if "ignore_pretraining_limits" in sig.parameters:
                kwargs["ignore_pretraining_limits"] = self.ignore_pretraining_limits
        except Exception:
            pass

        return TabPFNClassifier(**kwargs)

    def _get_embeddings(self, clf, X: np.ndarray) -> np.ndarray:
        """
        Extract internal (pre-output-head) representations from a fitted
        TabPFNClassifier.

        Requires TabPFN >= 2.0 (Prior Labs).  If get_embeddings() is not
        available, raises a clear RuntimeError with installation instructions.
        """
        if not hasattr(clf, "get_embeddings"):
            raise RuntimeError(
                "TabPFN version does not expose get_embeddings().\n"
                "Please install tabpfn >= 2.0:\n"
                "    pip install tabpfn\n"
                "See README_TabPFN.md for offline installation instructions."
            )
        emb = clf.get_embeddings(X)
        return np.asarray(emb, dtype=np.float32)

    def _apply_context_limit(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optionally subsample (X, y) to at most max_context_samples rows.
        Uses a deterministic random state for reproducibility.
        """
        if self.max_context_samples is not None and len(X) > self.max_context_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=self.max_context_samples, replace=False)
            return X[idx], y[idx]
        return X, y

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_train_embeddings_oof(
        self, X: np.ndarray, y_dummy: np.ndarray
    ) -> np.ndarray:
        """
        Out-of-fold embedding extraction for *training* samples.

        For each stratified K-fold split:
          - Fit TabPFN on the context folds (optionally subsampled).
          - Call get_embeddings() on the query fold.

        This prevents a training sample's label from influencing its own
        embedding (analogous to the train-time leakage problem in kNN).

        Returns
        -------
        embeddings : np.ndarray of shape (n_train, embed_dim)
        """
        n = len(X)
        embeddings: Optional[np.ndarray] = None

        kf = StratifiedKFold(
            n_splits=self.n_oof_splits, shuffle=True, random_state=self.random_state
        )

        for fold_idx, (ctx_idx, qry_idx) in enumerate(kf.split(X, y_dummy)):
            if self.verbose:
                n_pos = y_dummy[ctx_idx].sum()
                print(
                    f"  [OOF {fold_idx + 1}/{self.n_oof_splits}] "
                    f"context={len(ctx_idx)} (pos={n_pos}), "
                    f"query={len(qry_idx)}"
                )

            X_ctx, y_ctx = self._apply_context_limit(X[ctx_idx], y_dummy[ctx_idx])

            clf_fold = self._make_tabpfn()
            clf_fold.fit(X_ctx, y_ctx)

            emb_fold = self._get_embeddings(clf_fold, X[qry_idx])

            if embeddings is None:
                embeddings = np.zeros((n, emb_fold.shape[1]), dtype=np.float32)
            embeddings[qry_idx] = emb_fold

        return embeddings  # type: ignore[return-value]

    def _extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Embed X using the full-context TabPFN model stored during fit().
        Used for validation and test inference.
        """
        check_is_fitted(self, "tabpfn_model_")
        return self._get_embeddings(self.tabpfn_model_, X)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Tuple[Iterable, Iterable],
        X_val: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
        y_val: Optional[Tuple[Iterable, Iterable]] = None,
    ) -> "TabPFNCompetingRisks":
        """
        Fit K cause-specific Cox heads on frozen TabPFN embeddings.

        Parameters
        ----------
        X : array-like of shape (n_train, n_features)
        y : tuple (durations, event_types)
            durations : float array — time to event or censoring.
            event_types : int array — 0=censored, 1…K=event types.
        X_val, y_val : optional validation data for early stopping.

        Returns
        -------
        self
        """
        # ---- sklearn validation ----
        if OLD_SKLEARN:
            X = self._validate_data(X, dtype=None, cast_to_ndarray=False, reset=True)
        else:
            X = self._validate_data(X, dtype=None, skip_check_array=True, reset=True)
        X = np.asarray(X, dtype=np.float32)

        durations, event_types = y
        durations = np.asarray(durations, dtype=float)
        event_types = np.asarray(event_types, dtype=int)

        unique_events = np.unique(event_types)
        if unique_events.min() < 0:
            raise ValueError("Event types must be >= 0 (0=censored, 1…K=events).")
        if unique_events.max() > self.n_event_types:
            raise ValueError(
                f"Max event type {unique_events.max()} > n_event_types={self.n_event_types}."
            )

        # ---- device ----
        if self.device is None:
            self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(
                "cuda:0" if self.device == "cuda" else self.device
            )
        else:
            self.device_ = self.device

        # ---- build dummy labels for TabPFN context ----
        # Binary: any event = 1, censored = 0.
        y_dummy = (event_types > 0).astype(int)

        # ---- fit full-context TabPFN (for val/test embedding) ----
        if self.verbose:
            print("Fitting full-context TabPFN model (val/test encoder)...")
        X_ctx, y_ctx = self._apply_context_limit(X, y_dummy)
        self.tabpfn_model_ = self._make_tabpfn()
        self.tabpfn_model_.fit(X_ctx, y_ctx)
        self.tabpfn_context_X_ = X_ctx  # stored for inspection
        self.tabpfn_context_y_ = y_ctx

        # ---- OOF embeddings for training ----
        if self.verbose:
            print(
                f"Extracting OOF training embeddings "
                f"({self.n_oof_splits} folds)..."
            )
        embeddings = self._extract_train_embeddings_oof(X, y_dummy)
        embed_dim = embeddings.shape[1]

        if self.verbose:
            print(f"TabPFN embedding dimension: {embed_dim}")

        # ---- validation embeddings ----
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            durations_val, event_types_val = y_val
            durations_val = np.asarray(durations_val, dtype=float)
            event_types_val = np.asarray(event_types_val, dtype=int)
            if self.verbose:
                print("Extracting validation embeddings...")
            embeddings_val = self._extract_embeddings(X_val)
        else:
            embeddings_val = None
            durations_val = None
            event_types_val = None

        # ---- train K cause-specific Cox heads ----
        self.models_: List[nn.Module] = []
        self.baseline_cum_hazards_: List[np.ndarray] = []
        self.event_times_per_cause_: List[np.ndarray] = []

        X_tensor = _to_tensor(embeddings, self.device_)
        t_tensor = _to_tensor(durations, self.device_)

        for k in range(1, self.n_event_types + 1):
            n_events_k = int((event_types == k).sum())
            if self.verbose:
                print(f"\n--- Cox head: event type {k}/{self.n_event_types} "
                      f"({n_events_k} events) ---")

            events_k = (event_types == k).astype(float)
            e_tensor = _to_tensor(events_k, self.device_)

            model_k = _MLPRisk(embed_dim, hidden=self.hidden, dropout=self.dropout).to(
                self.device_
            )
            opt_k = torch.optim.AdamW(
                model_k.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            best_val = math.inf
            best_state = None
            no_improve = 0

            rng = np.random.default_rng(self.random_state + k)
            idx = np.arange(len(embeddings))

            for epoch in range(self.epochs):
                rng.shuffle(idx)

                model_k.train()
                opt_k.zero_grad(set_to_none=True)
                risks = model_k(X_tensor[idx])
                loss = _coxph_breslow_loss(risks, t_tensor[idx], e_tensor[idx], None)
                loss.backward()
                opt_k.step()

                # ---- early stopping on validation ----
                if embeddings_val is not None:
                    model_k.eval()
                    with torch.no_grad():
                        ev_k = (event_types_val == k).astype(float)
                        rv = model_k(_to_tensor(embeddings_val, self.device_))
                        lv = _coxph_breslow_loss(
                            rv,
                            _to_tensor(durations_val, self.device_),
                            _to_tensor(ev_k, self.device_),
                            None,
                        ).item()

                    if lv + 1e-9 < best_val:
                        best_val = lv
                        best_state = {
                            key: val.detach().cpu().clone()
                            for key, val in model_k.state_dict().items()
                        }
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            if best_state is not None:
                                model_k.load_state_dict(best_state)
                            if self.verbose:
                                print(f"  Early stopping at epoch {epoch}.")
                            break

                if self.verbose and epoch % 20 == 0:
                    print(f"  Epoch {epoch:04d} | train_loss={loss.item():.5f}")

            # ---- Breslow baseline hazard ----
            model_k.eval()
            with torch.no_grad():
                r_train_k = model_k(X_tensor).detach().cpu().numpy()

            etimes_k, H0_k = _breslow_baseline_hazard(r_train_k, durations, events_k)
            self.models_.append(model_k)
            self.baseline_cum_hazards_.append(H0_k)
            self.event_times_per_cause_.append(etimes_k)

        all_event_times = np.concatenate(self.event_times_per_cause_)
        self.event_times_ = np.unique(all_event_times)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_cause_specific_risk(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        cause: Optional[int] = None,
    ) -> np.ndarray:
        """
        Predict log cause-specific hazard scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        cause : int or None
            If int in [1, K], return risk for that cause — shape (n_samples,).
            If None, return all K risks — shape (n_samples, K).
        """
        check_is_fitted(self, "models_")

        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)
        X = np.asarray(X, dtype=np.float32)

        embeddings = self._extract_embeddings(X)
        X_tensor = _to_tensor(embeddings, self.device_)

        risks = []
        for model_k in self.models_:
            model_k.eval()
            with torch.no_grad():
                risks.append(model_k(X_tensor).detach().cpu().numpy())

        risks_arr = np.stack(risks, axis=1)  # (n_samples, K)

        if cause is not None:
            if not (1 <= cause <= self.n_event_types):
                raise ValueError(
                    f"cause must be in [1, {self.n_event_types}], got {cause}."
                )
            return risks_arr[:, cause - 1]

        return risks_arr

    def predict_cumulative_incidence(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        times: Optional[Iterable[float]] = None,
        return_array: bool = False,
        cause: Optional[int] = None,
    ) -> Union[
        List[Tuple[np.ndarray, np.ndarray]],
        np.ndarray,
        Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
        Dict[int, np.ndarray],
    ]:
        """
        Predict cause-specific cumulative incidence functions (CIF).

        CIF_k(t) = P(T ≤ t, cause = k) computed via:
            CIF_k(t) = ∫_0^t S(u-) dH_k(u)
        where S(t) = exp(-Σ_j H_j(t)) is the all-cause survival.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like or None
            Evaluation time points.  Defaults to all training event times.
        return_array : bool
            True → return np.ndarray of shape (n_samples, n_times).
            False → return list of (times, CIF) tuples per sample.
        cause : int or None
            Specific cause or all causes (None).
        """
        check_is_fitted(
            self,
            ["models_", "baseline_cum_hazards_", "event_times_per_cause_", "event_times_"],
        )

        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)
        X = np.asarray(X, dtype=np.float32)

        embeddings = self._extract_embeddings(X)
        X_tensor = _to_tensor(embeddings, self.device_)
        n_samples = embeddings.shape[0]

        if times is None:
            times = self.event_times_
        times = np.sort(np.asarray(times, dtype=float))
        n_times = len(times)

        # Collect risk scores for all causes
        all_risks: List[np.ndarray] = []
        for model_k in self.models_:
            model_k.eval()
            with torch.no_grad():
                all_risks.append(model_k(X_tensor).detach().cpu().numpy())

        def _cif_for_cause(cause_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            cif_samples = []
            for i in range(n_samples):
                H_all = np.zeros((self.n_event_types, n_times))
                for k_idx in range(self.n_event_types):
                    H0 = self.baseline_cum_hazards_[k_idx]
                    etimes = self.event_times_per_cause_[k_idx]
                    H0_at_t = np.interp(
                        times, etimes, H0, left=0.0, right=H0[-1] if len(H0) else 0.0
                    )
                    H_all[k_idx] = H0_at_t * np.exp(all_risks[k_idx][i])

                H_total = H_all.sum(axis=0)
                S_t = np.exp(-H_total)

                H_k = H_all[cause_idx]
                dH_k = np.diff(np.concatenate([[0.0], H_k]))
                S_left = np.concatenate([[1.0], S_t])[:-1]
                cif = np.cumsum(S_left * dH_k)
                cif_samples.append(cif)

            return times, np.array(cif_samples)  # (n_samples, n_times)

        if cause is not None:
            if not (1 <= cause <= self.n_event_types):
                raise ValueError(
                    f"cause must be in [1, {self.n_event_types}], got {cause}."
                )
            t_out, cif_arr = _cif_for_cause(cause - 1)
            if return_array:
                return cif_arr
            return [(t_out, cif_arr[i]) for i in range(n_samples)]

        result: Dict = {}
        for k in range(1, self.n_event_types + 1):
            t_out, cif_arr = _cif_for_cause(k - 1)
            result[k] = cif_arr if return_array else [(t_out, cif_arr[i]) for i in range(n_samples)]
        return result

    def score(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Tuple[Iterable, Iterable],
        cause: Optional[int] = None,
    ) -> float:
        """
        Cause-specific concordance index.

        Parameters
        ----------
        X : array-like
        y : (durations, event_types)
        cause : int or None
            Specific cause → C-index for that cause.
            None → weighted average across all causes (weight = n_events_k).

        Returns
        -------
        c_index : float  (0.5 = random, 1.0 = perfect)
        """
        durations, event_types = y
        durations = np.asarray(durations, dtype=float)
        event_types = np.asarray(event_types, dtype=int)

        if cause is not None:
            events_k = (event_types == cause).astype(float)
            if events_k.sum() == 0:
                return float("nan")
            return _concordance_index(
                durations, events_k, self.predict_cause_specific_risk(X, cause=cause)
            )

        c_indices, weights = [], []
        for k in range(1, self.n_event_types + 1):
            events_k = (event_types == k).astype(float)
            n_k = int(events_k.sum())
            if n_k > 0:
                c_indices.append(
                    _concordance_index(
                        durations, events_k, self.predict_cause_specific_risk(X, cause=k)
                    )
                )
                weights.append(n_k)

        if not c_indices:
            return float("nan")

        weights_arr = np.array(weights, dtype=float) / sum(weights)
        return float(np.average(c_indices, weights=weights_arr))


# ---------------------------------------------------------------------------
# Sklearn >= 1.6 tags
# ---------------------------------------------------------------------------
if not OLD_SKLEARN:
    def _sklearn_tags_method(self):
        tags = super(TabPFNCompetingRisks, self).__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    TabPFNCompetingRisks.__sklearn_tags__ = _sklearn_tags_method
