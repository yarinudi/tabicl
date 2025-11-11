from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from packaging import version
import sklearn

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

if TYPE_CHECKING:
    import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
OLD_SKLEARN = version.parse(sklearn.__version__) < version.parse("1.6")


# -------------------------------
# Utilities
# -------------------------------

def _to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    x = np.asarray(x)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _coxph_breslow_loss(
    risks: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cox partial log-likelihood with Breslow ties.

    Parameters
    ----------
    risks : (n,) tensor
        Predicted log-risk scores (higher => higher hazard).
    durations : (n,) tensor
        Observed times.
    events : (n,) tensor in {0,1}
        Event indicators (1 = event observed, 0 = censored).
    sample_weight : (n,) tensor or None
        Optional per-sample weights.

    Returns
    -------
    loss : scalar tensor
        Negative partial log-likelihood (to minimize).

    Reference
    ---------
    Breslow, N. (1974). Covariance analysis of censored survival data.
    Biometrika, 51(3/4), 557-565.
    """
    # Sort by decreasing duration so risk sets are cumulative sums going forward
    order = torch.argsort(durations, descending=True)
    dur = durations[order]
    evt = events[order]
    r = risks[order]

    if sample_weight is None:
        sw = torch.ones_like(evt, dtype=r.dtype, device=r.device)
    else:
        sw = sample_weight[order].to(r.dtype)

    # Compute cumulative log-sum-exp over risk for the risk set at each time
    # log \sum_{j in R_i} exp(r_j) can be computed as logcumsumexp over sorted times
    log_cumsum_exp_r = torch.logcumsumexp(r, dim=0)  # (n,)

    # For Breslow ties, group by unique time among events
    # We compute: sum_{i: event} [ r_i - log sum_{j in R_i} exp(r_j) ]
    # If multiple events at same time t, denominator is raised to the number of ties.
    _, _, counts = torch.unique_consecutive(dur, return_inverse=True, return_counts=True)

    # Mask events only
    event_mask = evt > 0
    if not torch.any(event_mask):
        return torch.zeros([], dtype=r.dtype, device=r.device)

    # Accumulate per-observation loglik terms
    # First term: sum w_i * r_i over events
    term1 = (sw * r * event_mask.to(r.dtype)).sum()

    # Second term: sum over event times of log risk set weighted by number of events at that time
    # We need, for each position k which is the first index of a block of equal times,
    # the logcumsumexp at that k, multiplied by (#events at time t).
    # Find start indices of each block in the sorted order
    # positions k = 0, c0, c0+c1, ...
    block_starts = torch.cumsum(torch.cat([torch.tensor([0], device=r.device), counts[:-1]]), dim=0)
    # For each block, compute how many events in the block
    # (weights only for event rows in the block). We'll use integer counts if no weights.
    block_events = []
    for start, cnt in zip(block_starts.tolist(), counts.tolist()):
        sl = slice(start, start + cnt)
        # Sum of sample weights for events at this time
        be = (sw[sl] * evt[sl].to(r.dtype)).sum()
        block_events.append(be)
    block_events = torch.stack(block_events)  # (n_unique,)

    # Risk set log-sum-exp evaluated at block starts
    denom_terms = log_cumsum_exp_r[block_starts]

    term2 = (block_events * denom_terms).sum()

    # Negative partial log-likelihood
    npll = -(term1 - term2) / sw.sum()
    return npll


def _breslow_baseline_hazard(
    risks: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Breslow baseline cumulative hazard H0(t).

    Returns
    -------
    unique_event_times : array, shape (m,)
    cum_h0 : array, shape (m,)
    """
    order = np.argsort(durations)
    dur_sorted = durations[order]
    e = events[order].astype(bool)
    r = np.exp(risks[order])

    unique_times, idx_start, counts = np.unique(dur_sorted, return_index=True, return_counts=True)

    # cumulative sums over risk set (from right)
    # risk set at time i includes i..end
    rev_cumsum_r = np.cumsum(r[::-1])[::-1]

    h0_increments = []
    for s, c in zip(idx_start, counts):
        # number of events at this time
        d_t = e[s:s + c].sum()
        if d_t == 0:
            h0_increments.append(0.0)
            continue
        # denominator is sum of risks over risk set at first index of this time block
        denom = rev_cumsum_r[s]
        h0_increments.append(float(d_t) / float(denom))

    h0_increments = np.asarray(h0_increments)
    cum_h0 = np.cumsum(h0_increments)
    return unique_times, cum_h0


def _concordance_index(
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> float:
    """Harrell's C-index (time-independent) with tie handling.

    Higher C = better (risk higher means earlier event).
    """
    n = len(durations)
    assert durations.shape == events.shape == risks.shape
    # Comparable pairs: i has event (e_i=1), j has t_j >= t_i
    # Sort by durations to establish temporal ordering
    order = np.argsort(durations)
    e = events[order]
    r = risks[order]

    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    for i in range(n):
        if e[i] == 0:
            continue
        # Compare with j > i such that t_j >= t_i (already true by sort)
        ri = r[i]
        for j in range(i + 1, n):
            permissible += 1
            rj = r[j]
            if ri > rj:
                concordant += 1
            elif ri == rj:
                ties += 1
    if permissible == 0:
        return float("nan")
    return (concordant + 0.5 * ties) / permissible


# -------------------------------
# Models
# -------------------------------

class _MLPRisk(nn.Module):
    def __init__(self, in_dim: int, hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None or hidden <= 0:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -------------------------------
# Estimator
# -------------------------------

class TabICLSurvivor(BaseEstimator):
    """Survival model with a CoxPH loss and TabICL-style interface.

    This class is designed to live inside the TabICL repo under
    `src/tabicl/sklearn/survivor.py` and conform to the same public standards as
    `TabICLClassifier` (scikit-learn API, device handling, auto-download of
    checkpoints, lightweight `fit`, and verbose controls).

    Parameters
    ----------
    # Core TabICL knobs (used when backbone="tabicl")
    n_estimators : int, default=1
        Number of estimators for ensemble when using TabICL backbone.
        Note: Unlike TabICLClassifier, default is 1 for survival to reduce complexity.
    
    norm_methods : list[str] | None, default=None
        Normalization methods to apply. When None, uses ["none", "power"].
    
    feat_shuffle_method : str, default="none"
        Feature permutation strategy: 'none', 'shift', 'random', 'latin'.
        Default is 'none' for survival to keep it simple.
    
    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping.
    
    batch_size : int, default=8
        Batch size for TabICL inference (when backbone="tabicl").
    
    use_amp : bool, default=True
        Whether to use automatic mixed precision.
    
    model_path : str | None, default=None
        Path to the pre-trained TabICL checkpoint file.
    
    allow_auto_download : bool, default=True
        Whether to allow automatic download of checkpoints.
    
    checkpoint_version : str, default="tabicl-classifier-v1.1-0506.ckpt"
        Which TabICL checkpoint version to use.
    
    device : str | torch.device | None, default=None
        Device to use for inference. If None, defaults to CUDA if available, else CPU.
    
    random_state : int | None, default=42
        Random seed for reproducibility.
    
    n_jobs : int | None, default=None
        Number of threads to use for PyTorch on CPU.
    
    verbose : bool, default=False
        Whether to print detailed information during training.
    
    inference_config : dict | None, default=None
        Advanced inference configuration for TabICL components.

    # Survival head knobs
    hidden : int | None, default=None
        Hidden layer size for the Cox head. If None or <=0, uses linear model.
    
    dropout : float, default=0.0
        Dropout rate for the Cox head.
    
    lr : float, default=1e-3
        Learning rate for training the Cox head.
    
    weight_decay : float, default=1e-4
        Weight decay for AdamW optimizer.
    
    epochs : int, default=200
        Maximum number of training epochs.
    
    patience : int, default=20
        Early stopping patience (number of epochs without improvement).

    backbone : {"mlp", "tabicl"}, default="tabicl"
        When "tabicl", loads TabICL checkpoint and uses it as frozen feature extractor.
        When "mlp", operates directly on standardized numeric features.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the training data.
    
    model_ : nn.Module
        The fitted Cox risk model.
    
    x_mean_ : np.ndarray
        Feature means (used for standardization when backbone="mlp").
    
    x_std_ : np.ndarray
        Feature standard deviations (used when backbone="mlp").
    
    event_times_ : np.ndarray
        Unique event times from training data.
    
    baseline_cum_hazard_ : np.ndarray
        Baseline cumulative hazard at each event time.
    
    device_ : torch.device
        The device where computations are performed.
    
    tabicl_model_ : TabICL | None
        The loaded TabICL model (when backbone="tabicl").
    
    X_encoder_ : TransformToNumerical | None
        Encoder for transforming input features to numerical values.

    Notes
    -----
    * `fit` accepts `(durations, events)` in `y` (sklearn style) or via
      explicit keyword args.
    * When `backbone="tabicl"`, this class loads the TabICL checkpoint and uses
      it as a frozen feature extractor to generate embeddings for survival modeling.
    * The TabICL model is never fine-tuned; only the Cox head is trained.
    """

    def __init__(
        self,
        # TabICL-compat params
        n_estimators: int = 1,
        norm_methods: Optional[List[str]] = None,
        feat_shuffle_method: str = "none",
        outlier_threshold: float = 4.0,
        batch_size: int = 8,
        use_amp: bool = True,
        model_path: Optional[str | Path] = None,
        allow_auto_download: bool = True,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        device: Optional[str | torch.device] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[Dict[str, Any]] = None,
        # Survival head params
        hidden: Optional[int] = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        patience: int = 20,
        # Backbone toggle
        backbone: str = "tabicl",
    ):
        # Store TabICL config for parity
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.n_jobs = n_jobs
        self.inference_config = inference_config
        self.verbose = verbose

        # Survival head
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

        # Random state
        self.random_state = random_state

        # Backbone
        self.backbone = backbone

    def _more_tags(self):
        """Mark estimator as non-deterministic to bypass certain sklearn tests."""
        return dict(non_deterministic=True)

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _load_tabicl_model(self):
        """
        Load TabICL model with fallback to local checkpoints directory.
        
        Strategy:
        1. Try to load from HuggingFace Hub (default)
        2. If fails, fallback to local 'tabicl_checkpoints' directory
        3. When loaded from HuggingFace, save copy to 'tabicl_checkpoints'
        4. Auto-decompress .xz files if needed
        """
        from ..model.tabicl import TabICL
        import shutil
        import lzma
        
        repo_id = "jingang/TabICL-clf"
        filename = self.checkpoint_version

        # Local checkpoints directory
        local_checkpoints_dir = Path("tabicl_checkpoints")
        local_checkpoint_path = local_checkpoints_dir / filename
        local_checkpoint_compressed = local_checkpoints_dir / f"{filename}.xz"

        ckpt_legacy = "tabicl-classifier.ckpt"
        ckpt_v1 = "tabicl-classifier-v1-0208.ckpt"
        ckpt_v1_1 = "tabicl-classifier-v1.1-0506.ckpt"

        if filename == ckpt_legacy:
            info_message = (
                f"INFO: You are using '{ckpt_legacy}'. This is a legacy alias for '{ckpt_v1}' "
                f"and is maintained for backward compatibility. It may be removed in a future release.\n"
                f"Please consider using '{ckpt_v1}' or the latest '{ckpt_v1_1}' directly.\n"
            )
        elif filename == ckpt_v1:
            info_message = (
                f"INFO: You are downloading '{ckpt_v1}', the version used in the original TabICL paper.\n"
                f"A newer version, '{ckpt_v1_1}', is available and offers improved performance.\n"
            )
        elif filename == ckpt_v1_1:
            info_message = (
                f"INFO: You are downloading '{ckpt_v1_1}', the latest best-performing version of TabICL.\n"
            )
        else:
            raise ValueError(
                f"Invalid checkpoint version '{filename}'. Available ones are: '{ckpt_legacy}', '{ckpt_v1}', '{ckpt_v1_1}'."
            )

        if self.model_path is None:
            # Try HuggingFace first, fallback to local checkpoints directory
            try:
                # Try to load from HuggingFace cache
                model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
                if self.verbose:
                    print(f"Loaded checkpoint from HuggingFace cache: {model_path_}")
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                
                # Save copy to local checkpoints directory
                try:
                    local_checkpoints_dir.mkdir(parents=True, exist_ok=True)
                    if not local_checkpoint_path.exists():
                        shutil.copy2(model_path_, local_checkpoint_path)
                        if self.verbose:
                            print(f"Saved copy to: {local_checkpoint_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not save to local checkpoints: {e}")
                
            except (LocalEntryNotFoundError, Exception) as e:
                # Try local checkpoints directory
                if local_checkpoint_path.exists():
                    if self.verbose:
                        print(f"Loading checkpoint from local directory: {local_checkpoint_path}")
                    model_path_ = local_checkpoint_path
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                elif local_checkpoint_compressed.exists():
                    # Decompress .xz file
                    if self.verbose:
                        print(f"Found compressed checkpoint: {local_checkpoint_compressed}")
                        print(f"Decompressing to: {local_checkpoint_path}")
                    try:
                        with lzma.open(local_checkpoint_compressed, 'rb') as f_in:
                            with open(local_checkpoint_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        if self.verbose:
                            print("Decompression complete!")
                        model_path_ = local_checkpoint_path
                        checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                    except Exception as decompress_err:
                        raise ValueError(
                            f"Failed to decompress checkpoint.\n"
                            f"Error: {decompress_err}\n"
                            f"Try deleting '{local_checkpoint_compressed}' and re-downloading."
                        )
                elif self.allow_auto_download:
                    # Download from HuggingFace
                    if self.verbose:
                        print(info_message)
                        print(f"Checkpoint '{filename}' not cached.\nDownloading from Hugging Face Hub ({repo_id}).\n")
                    
                    try:
                        model_path_ = Path(hf_hub_download(repo_id=repo_id, filename=filename))
                        checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                        
                        # Save to local checkpoints directory
                        try:
                            local_checkpoints_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(model_path_, local_checkpoint_path)
                            if self.verbose:
                                print(f"Saved checkpoint to: {local_checkpoint_path}")
                        except Exception as save_err:
                            if self.verbose:
                                print(f"Warning: Could not save to local checkpoints: {save_err}")
                    except Exception as download_err:
                        raise ValueError(
                            f"Failed to download checkpoint from HuggingFace.\n"
                            f"Error: {download_err}\n"
                            f"Please ensure you have internet connection or place checkpoint in '{local_checkpoints_dir}'."
                        )
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not found.\n"
                        f"Checked: HuggingFace cache and '{local_checkpoint_path}'.\n"
                        f"Set allow_auto_download=True to download, or place checkpoint in '{local_checkpoints_dir}'."
                    )
        else:
            # Use provided path
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                if self.allow_auto_download:
                    if self.verbose:
                        print(info_message)
                        print(
                            f"Checkpoint not found at '{model_path_}'.\n"
                            f"Downloading '{filename}' from Hugging Face Hub ({repo_id}) to this location.\n"
                        )
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path_}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True."
                    )

        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

        self.model_path_ = model_path_
        tabicl_model = TabICL(**checkpoint["config"])
        tabicl_model.load_state_dict(checkpoint["state_dict"])
        tabicl_model.eval()
        
        # Freeze TabICL model
        for param in tabicl_model.parameters():
            param.requires_grad = False
        
        return tabicl_model

    def _extract_tabicl_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from TabICL model.
        
        Uses the TabICL model to generate row representations that capture
        feature interactions. These are used as input to the Cox head.
        """
        check_is_fitted(self, ["tabicl_model_", "X_encoder_", "inference_config_"])
        
        # Transform to numerical
        X_num = self.X_encoder_.transform(X)
        
        # Convert to tensor
        X_tensor = _to_tensor(X_num, self.device_)
        
        # Add batch dimension if needed: (n_samples, n_features) -> (1, n_samples, n_features)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(0)
        
        # Extract embeddings through TabICL's encoding layers
        # TabICL: col_embedder -> row_interactor -> ICL
        # For feature extraction, we only use col_embedder + row_interactor
        # IMPORTANT: Pass inference config to properly configure device handling
        with torch.no_grad():
            # Column embeddings: (1, n_samples, n_features) -> (1, n_samples, n_features + n_cls, embed_dim)
            col_embs = self.tabicl_model_.col_embedder(
                X_tensor,
                mgr_config=self.inference_config_.COL_CONFIG
            )
            # Row interactions: (1, n_samples, n_features + n_cls, embed_dim) -> (1, n_samples, n_cls * embed_dim)
            row_embs = self.tabicl_model_.row_interactor(
                col_embs,
                mgr_config=self.inference_config_.ROW_CONFIG
            )
            # Remove batch dimension: (1, n_samples, n_cls * embed_dim) -> (n_samples, n_cls * embed_dim)
            row_embs = row_embs.squeeze(0)
        
        return row_embs.cpu().numpy()

    def _standardize_inputs(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Standardize inputs for MLP backbone."""
        Xn = np.asarray(X)
        x_mean = getattr(self, 'x_mean_', None)
        if x_mean is None:
            self.x_mean_ = Xn.mean(axis=0)
            self.x_std_ = Xn.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
        Xs_out = (Xn - self.x_mean_) / self.x_std_
        return Xs_out

    # ---------------------------
    # Public API
    # ---------------------------
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Optional[Union[Tuple[Iterable, Iterable], np.ndarray]] = None,
        durations: Optional[Iterable[float]] = None,
        events: Optional[Iterable[int]] = None,
        sample_weight: Optional[Iterable[float]] = None,
        X_val: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        durations_val: Optional[Iterable[float]] = None,
        events_val: Optional[Iterable[int]] = None,
    ) -> TabICLSurvivor:
        """Fit the survival model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        
        y : tuple of (durations, events) or None
            Survival data as (durations, events). If provided, overrides
            separate durations and events arguments.
        
        durations : array-like of shape (n_samples,) or None
            Time to event or censoring for each sample.
        
        events : array-like of shape (n_samples,) or None
            Event indicator (1=event observed, 0=censored) for each sample.
        
        sample_weight : array-like of shape (n_samples,) or None
            Sample weights.
        
        X_val : array-like of shape (n_val_samples, n_features) or None
            Validation input data for early stopping.
        
        durations_val : array-like of shape (n_val_samples,) or None
            Validation durations.
        
        events_val : array-like of shape (n_val_samples,) or None
            Validation events.

        Returns
        -------
        self : TabICLSurvivor
            Fitted estimator.
        """
        # Validate data
        if OLD_SKLEARN:
            X = self._validate_data(X, dtype=None, cast_to_ndarray=False, reset=True)
        else:
            X = self._validate_data(X, dtype=None, skip_check_array=True, reset=True)
        
        # Parse survival targets
        if y is not None and (durations is None and events is None):
            durations, events = y  # type: ignore
        if durations is None or events is None:
            raise ValueError("Provide durations and events (either via y=(t,e) or separate args).")

        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=float)

        # Setup device (TabICL expects device with index, e.g. "cuda:0" not "cuda")
        if self.device is None:
            if torch.cuda.is_available():
                self.device_ = torch.device("cuda:0")
            else:
                self.device_ = torch.device("cpu")
        elif isinstance(self.device, str):
            # If user specified "cuda" without index, add index 0
            if self.device == "cuda":
                self.device_ = torch.device("cuda:0")
            else:
                self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

        # Prepare features based on backbone
        if self.backbone.lower() == "tabicl":
            # Load TabICL model and preprocessing
            from .preprocessing import TransformToNumerical
            from ..model.inference_config import InferenceConfig
            
            self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
            self.X_encoder_.fit(X)
            
            self.tabicl_model_ = self._load_tabicl_model()
            self.tabicl_model_.to(self.device_)
            self.tabicl_model_.eval()
            
            # Setup inference configuration for TabICL
            init_config = {
                "COL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
                "ROW_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
                "ICL_CONFIG": {"device": self.device_, "use_amp": self.use_amp, "verbose": self.verbose},
            }
            if self.inference_config is None:
                self.inference_config_ = InferenceConfig()
                self.inference_config_.update_from_dict(init_config)
            elif isinstance(self.inference_config, dict):
                self.inference_config_ = InferenceConfig()
                for key, value in self.inference_config.items():
                    if key in init_config:
                        init_config[key].update(value)
                self.inference_config_.update_from_dict(init_config)
            else:
                self.inference_config_ = self.inference_config
            
            # Extract embeddings
            Xs = self._extract_tabicl_embeddings(X)
        else:
            # MLP backbone: use standardized features
            self.X_encoder_ = None
            self.tabicl_model_ = None
            Xs = self._standardize_inputs(np.asarray(X))

        in_dim = Xs.shape[1]
        model = _MLPRisk(in_dim, hidden=self.hidden, dropout=self.dropout).to(self.device_)
        self.model_ = model

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val = math.inf
        best_state = None
        no_improve = 0

        X_tensor = _to_tensor(Xs, self.device_)
        t_tensor = _to_tensor(durations, self.device_)
        e_tensor = _to_tensor(events, self.device_)
        w_tensor = None if sample_weight is None else _to_tensor(np.asarray(sample_weight, dtype=float), self.device_)

        n = Xs.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx = np.arange(n)

        for epoch in range(self.epochs):
            rng.shuffle(idx)
            xb = X_tensor[idx]
            tb = t_tensor[idx]
            eb = e_tensor[idx]
            wb = None if w_tensor is None else w_tensor[idx]

            model.train()
            opt.zero_grad(set_to_none=True)
            risks = model(xb)
            loss = _coxph_breslow_loss(risks, tb, eb, wb)
            loss.backward()
            opt.step()

            if X_val is not None and durations_val is not None and events_val is not None:
                model.eval()
                with torch.no_grad():
                    if self.backbone.lower() == "tabicl":
                        Xv = self._extract_tabicl_embeddings(X_val)
                    else:
                        Xv = self._standardize_inputs(np.asarray(X_val))
                    rv = model(_to_tensor(Xv, self.device_))
                    lv = _coxph_breslow_loss(
                        rv,
                        _to_tensor(np.asarray(durations_val, dtype=float), self.device_),
                        _to_tensor(np.asarray(events_val, dtype=float), self.device_),
                        None,
                    ).item()
                if lv + 1e-9 < best_val:
                    best_val = lv
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if best_state is not None:
                            model.load_state_dict(best_state)
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:04d} | loss={loss.item():.5f}")

        # Compute baseline hazard
        with torch.no_grad():
            model.eval()
            r_train = model(X_tensor).detach().cpu().numpy()
        etimes, H0 = _breslow_baseline_hazard(r_train, durations, events)
        self.event_times_ = etimes
        self.baseline_cum_hazard_ = H0
        
        return self

    def predict_risk(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        """Predict risk scores (log-hazard).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        risks : np.ndarray of shape (n_samples,)
            Predicted risk scores (higher = higher hazard).
        """
        check_is_fitted(self, "model_")
        
        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)
        
        if self.backbone.lower() == "tabicl":
            Xs = self._extract_tabicl_embeddings(X)
        else:
            Xs = self._standardize_inputs(np.asarray(X))
        
        with torch.no_grad():
            risks = self.model_(_to_tensor(Xs, self.device_)).detach().cpu().numpy()
        return risks

    def predict_survival_function(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        times: Optional[Iterable[float]] = None,
        return_array: bool = False,
    ) -> Union[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Predict survival function S(t) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        times : array-like or None
            Time points at which to evaluate survival function.
            If None, uses training event times.
        
        return_array : bool, default=False
            If True, return array of shape (n_samples, n_times).
            If False, return list of (times, S) tuples.

        Returns
        -------
        survival_functions : list or np.ndarray
            If return_array=False: list of (times, S(times)) tuples.
            If return_array=True: array of shape (n_samples, n_times).
        """
        check_is_fitted(self, ["model_", "baseline_cum_hazard_", "event_times_"])
        
        risks = self.predict_risk(X)
        
        if times is None:
            times = self.event_times_
        times = np.asarray(times, dtype=float)
        
        # Interpolate baseline cumulative hazard
        H0_t = np.interp(times, self.event_times_, self.baseline_cum_hazard_, 
                        left=0, right=self.baseline_cum_hazard_[-1])
        
        out = []
        for r in risks:
            # S(t) = exp(-H0(t) * exp(risk))
            S = np.exp(-np.clip(H0_t * np.exp(r), a_min=None, a_max=1e6))
            out.append((times, S))
        
        if return_array:
            return np.stack([s for _, s in out], axis=0)
        return out

    def score(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame], 
              y: Tuple[Iterable, Iterable]) -> float:
        """Compute concordance index (C-index).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        y : tuple of (durations, events)
            True survival data.

        Returns
        -------
        c_index : float
            Harrell's concordance index (higher is better).
        """
        durations, events = y
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=float)
        risks = self.predict_risk(X)
        return _concordance_index(durations, events, risks)


# Add __sklearn_tags__ method for sklearn >= 1.6
if not OLD_SKLEARN:
    def _sklearn_tags_method(self):
        """Sklearn tags for compatibility with sklearn >= 1.6."""
        tags = super(TabICLSurvivor, self).__sklearn_tags__()
        tags.non_deterministic = True
        return tags
    
    TabICLSurvivor.__sklearn_tags__ = _sklearn_tags_method
