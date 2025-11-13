from __future__ import annotations

"""
TabICLCompetingRisks — Competing risks survival model with TabICL embeddings.

This class extends TabICLSurvivor to handle multiple competing event types using
cause-specific hazards approach. Each event type gets its own Cox model trained
on shared TabICL embeddings.

Author: (Your Name)
License: MIT
"""

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

# Import from survivor module
from .survivor import (
    _to_tensor,
    _coxph_breslow_loss,
    _breslow_baseline_hazard,
    _concordance_index,
    _MLPRisk,
)


class TabICLCompetingRisks(BaseEstimator):
    """
    Competing risks survival model using TabICL embeddings and cause-specific hazards.
    
    This model handles multiple competing event types by:
    - Extracting shared TabICL embeddings (frozen feature extractor)
    - Training K separate Cox models (one per event type)
    - Treating competing events as censored for each cause-specific model
    
    Parameters
    ----------
    n_event_types : int
        Number of competing event types (K).
        Event type 0 is always treated as censoring.
        Event types 1, 2, ..., K are the competing events.
    
    # TabICL parameters (same as TabICLSurvivor)
    n_estimators : int, default=1
        Number of ensemble members for TabICL.
    
    norm_methods : list[str] | None, default=None
        Normalization methods. When None, uses ["none", "power"].
    
    feat_shuffle_method : str, default="none"
        Feature permutation strategy.
    
    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection.
    
    batch_size : int, default=8
        Batch size for TabICL inference.
    
    use_amp : bool, default=True
        Whether to use automatic mixed precision.
    
    model_path : str | None, default=None
        Path to pretrained TabICL checkpoint.
    
    allow_auto_download : bool, default=True
        Whether to allow automatic checkpoint download.
    
    checkpoint_version : str, default="tabicl-classifier-v1.1-0506.ckpt"
        Which TabICL checkpoint to use.
    
    device : str | torch.device | None, default=None
        Device for inference. If None, auto-detects.
    
    random_state : int | None, default=42
        Random seed for reproducibility.
    
    n_jobs : int | None, default=None
        Number of CPU threads for PyTorch.
    
    verbose : bool, default=False
        Whether to print training progress.
    
    inference_config : dict | None, default=None
        Advanced inference configuration for TabICL.
    
    # Cox head parameters
    hidden : int | None, default=None
        Hidden layer size for Cox heads. If None, uses linear models.
    
    dropout : float, default=0.0
        Dropout rate for Cox heads.
    
    lr : float, default=1e-3
        Learning rate for training.
    
    weight_decay : float, default=1e-4
        Weight decay for AdamW optimizer.
    
    epochs : int, default=200
        Maximum training epochs per cause.
    
    patience : int, default=20
        Early stopping patience.
    
    backbone : str, default="tabicl"
        Feature extraction method: "tabicl" or "mlp".
    
    Attributes
    ----------
    n_features_in_ : int
        Number of input features.
    
    models_ : List[nn.Module]
        List of K fitted Cox models (one per event type).
    
    baseline_cum_hazards_ : List[np.ndarray]
        Baseline cumulative hazards for each event type.
    
    event_times_ : np.ndarray
        Unique event times across all causes.
    
    device_ : torch.device
        Device where computations are performed.
    
    tabicl_model_ : TabICL | None
        Loaded TabICL model (when backbone="tabicl").
    
    X_encoder_ : TransformToNumerical | None
        Feature encoder.
    
    inference_config_ : InferenceConfig | None
        TabICL inference configuration.
    
    Examples
    --------
    >>> import numpy as np
    >>> from tabicl import TabICLCompetingRisks
    >>> 
    >>> # Generate competing risks data
    >>> X = np.random.randn(500, 10)
    >>> times = np.random.exponential(100, 500)
    >>> event_types = np.random.choice([0, 1, 2], size=500, p=[0.3, 0.4, 0.3])
    >>> 
    >>> # Train model
    >>> model = TabICLCompetingRisks(
    ...     n_event_types=2,
    ...     backbone="tabicl",
    ...     hidden=128,
    ...     epochs=100
    ... )
    >>> model.fit(X[:400], (times[:400], event_types[:400]))
    >>> 
    >>> # Predict cause-specific risks
    >>> risks = model.predict_cause_specific_risk(X[400:])  # Shape: (100, 2)
    >>> 
    >>> # Evaluate cause-specific C-index
    >>> c_index_1 = model.score(X[400:], (times[400:], event_types[400:]), cause=1)
    >>> c_index_avg = model.score(X[400:], (times[400:], event_types[400:]))
    """
    
    def __init__(
        self,
        n_event_types: int,
        # TabICL params
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
        # Cox head params
        hidden: Optional[int] = None,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        patience: int = 20,
        # Backbone
        backbone: str = "tabicl",
    ):
        # Competing risks specific
        self.n_event_types = n_event_types
        
        # Store all params (same as TabICLSurvivor)
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
        
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.backbone = backbone
    
    def _more_tags(self):
        """Mark estimator as non-deterministic."""
        return dict(non_deterministic=True)
    
    # Reuse methods from TabICLSurvivor by copying implementation
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
        local_checkpoints_dir = Path("tabicl/tabicl_checkpoints")
        local_checkpoint_path = local_checkpoints_dir / filename
        local_checkpoint_compressed = local_checkpoints_dir / f"{filename}.xz"
        
        ckpt_legacy = "tabicl-classifier.ckpt"
        ckpt_v1 = "tabicl-classifier-v1-0208.ckpt"
        ckpt_v1_1 = "tabicl-classifier-v1.1-0506.ckpt"
        
        if filename == ckpt_legacy:
            info_message = (
                f"INFO: Using '{ckpt_legacy}' (legacy alias for '{ckpt_v1}'). "
                f"Consider using '{ckpt_v1_1}' for better performance.\n"
            )
        elif filename == ckpt_v1:
            info_message = (
                f"INFO: Using '{ckpt_v1}'. Newer version '{ckpt_v1_1}' available.\n"
            )
        elif filename == ckpt_v1_1:
            info_message = f"INFO: Using '{ckpt_v1_1}', the latest version.\n"
        else:
            raise ValueError(f"Invalid checkpoint version '{filename}'.")
        
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
                
            except (LocalEntryNotFoundError, Exception):
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
                        print(f"Downloading '{filename}' from Hugging Face Hub.\n")
                    
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
                        f"Enable allow_auto_download=True to download, or place checkpoint in '{local_checkpoints_dir}'."
                    )
        else:
            # User provided explicit path
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if model_path_.exists():
                checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
            else:
                if self.allow_auto_download:
                    if self.verbose:
                        print(f"Downloading to '{model_path_}'.\n")
                    model_path_.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path_.parent)
                    Path(cache_path).rename(model_path_)
                    checkpoint = torch.load(model_path_, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(f"Checkpoint not found at '{model_path_}'.")
        
        assert "config" in checkpoint and "state_dict" in checkpoint
        
        self.model_path_ = model_path_
        tabicl_model = TabICL(**checkpoint["config"])
        tabicl_model.load_state_dict(checkpoint["state_dict"])
        tabicl_model.eval()
        
        for param in tabicl_model.parameters():
            param.requires_grad = False
        
        return tabicl_model
    
    def _extract_tabicl_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract TabICL embeddings (same as TabICLSurvivor)."""
        check_is_fitted(self, ["tabicl_model_", "X_encoder_", "inference_config_"])
        
        X_num = self.X_encoder_.transform(X)
        X_tensor = _to_tensor(X_num, self.device_)
        
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(0)
        
        with torch.no_grad():
            col_embs = self.tabicl_model_.col_embedder(
                X_tensor,
                mgr_config=self.inference_config_.COL_CONFIG
            )
            row_embs = self.tabicl_model_.row_interactor(
                col_embs,
                mgr_config=self.inference_config_.ROW_CONFIG
            )
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
    
    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        y: Union[Tuple[Iterable, Iterable], np.ndarray],
        X_val: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        y_val: Optional[Union[Tuple[Iterable, Iterable], np.ndarray]] = None,
    ) -> TabICLCompetingRisks:
        """
        Fit K cause-specific Cox models.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        
        y : tuple of (durations, event_types)
            durations : array of shape (n_samples,)
                Time to event or censoring.
            event_types : array of shape (n_samples,)
                Event type: 0 = censored, 1, 2, ..., K = event types.
        
        X_val, y_val : validation data (optional)
            For early stopping.
        
        Returns
        -------
        self : TabICLCompetingRisks
            Fitted estimator.
        """
        # Validate data
        if OLD_SKLEARN:
            X = self._validate_data(X, dtype=None, cast_to_ndarray=False, reset=True)
        else:
            X = self._validate_data(X, dtype=None, skip_check_array=True, reset=True)
        
        durations, event_types = y
        durations = np.asarray(durations, dtype=float)
        event_types = np.asarray(event_types, dtype=int)
        
        # Validate event types
        unique_events = np.unique(event_types)
        if unique_events.min() < 0:
            raise ValueError("Event types must be non-negative (0=censored, 1,2,...,K=events)")
        if unique_events.max() > self.n_event_types:
            raise ValueError(
                f"Max event type {unique_events.max()} exceeds n_event_types={self.n_event_types}"
            )
        
        # Setup device
        if self.device is None:
            if torch.cuda.is_available():
                self.device_ = torch.device("cuda:0")
            else:
                self.device_ = torch.device("cpu")
        elif isinstance(self.device, str):
            if self.device == "cuda":
                self.device_ = torch.device("cuda:0")
            else:
                self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device
        
        # Extract features based on backbone
        if self.backbone.lower() == "tabicl":
            from .preprocessing import TransformToNumerical
            from ..model.inference_config import InferenceConfig
            
            self.X_encoder_ = TransformToNumerical(verbose=self.verbose)
            self.X_encoder_.fit(X)
            
            self.tabicl_model_ = self._load_tabicl_model()
            self.tabicl_model_.to(self.device_)
            self.tabicl_model_.eval()
            
            # Setup inference config
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
            
            # Extract embeddings ONCE (shared across all causes)
            embeddings = self._extract_tabicl_embeddings(X)
        else:
            # MLP backbone
            self.X_encoder_ = None
            self.tabicl_model_ = None
            embeddings = self._standardize_inputs(np.asarray(X))
        
        # Extract validation embeddings if provided
        if X_val is not None and y_val is not None:
            durations_val, event_types_val = y_val
            durations_val = np.asarray(durations_val, dtype=float)
            event_types_val = np.asarray(event_types_val, dtype=int)
            
            if self.backbone.lower() == "tabicl":
                embeddings_val = self._extract_tabicl_embeddings(X_val)
            else:
                embeddings_val = self._standardize_inputs(np.asarray(X_val))
        else:
            embeddings_val = None
            durations_val = None
            event_types_val = None
        
        # Train K cause-specific models
        self.models_ = []
        self.baseline_cum_hazards_ = []
        event_times_per_cause = []
        
        in_dim = embeddings.shape[1]
        
        for k in range(1, self.n_event_types + 1):
            if self.verbose:
                n_events_k = (event_types == k).sum()
                print(f"\nTraining cause-specific Cox model for event type {k} ({n_events_k} events)...")
            
            # Create cause-specific binary outcome
            # Event k vs censored/competing (treat competing as censored)
            events_k = (event_types == k).astype(float)
            
            # Train Cox model for this cause
            model_k = _MLPRisk(in_dim, hidden=self.hidden, dropout=self.dropout).to(self.device_)
            opt_k = torch.optim.AdamW(model_k.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            
            best_val = math.inf
            best_state = None
            no_improve = 0
            
            X_tensor = _to_tensor(embeddings, self.device_)
            t_tensor = _to_tensor(durations, self.device_)
            e_tensor = _to_tensor(events_k, self.device_)
            
            rng = np.random.default_rng(self.random_state)
            idx = np.arange(len(embeddings))
            
            for epoch in range(self.epochs):
                rng.shuffle(idx)
                xb = X_tensor[idx]
                tb = t_tensor[idx]
                eb = e_tensor[idx]
                
                model_k.train()
                opt_k.zero_grad(set_to_none=True)
                risks = model_k(xb)
                loss = _coxph_breslow_loss(risks, tb, eb, None)
                loss.backward()
                opt_k.step()
                
                # Validation
                if embeddings_val is not None:
                    model_k.eval()
                    with torch.no_grad():
                        events_k_val = (event_types_val == k).astype(float)
                        rv = model_k(_to_tensor(embeddings_val, self.device_))
                        lv = _coxph_breslow_loss(
                            rv,
                            _to_tensor(durations_val, self.device_),
                            _to_tensor(events_k_val, self.device_),
                            None,
                        ).item()
                    
                    if lv + 1e-9 < best_val:
                        best_val = lv
                        best_state = {k: v.detach().cpu().clone() for k, v in model_k.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            if best_state is not None:
                                model_k.load_state_dict(best_state)
                            if self.verbose:
                                print(f"  Early stopping at epoch {epoch}")
                            break
                
                if self.verbose and epoch % 20 == 0:
                    print(f"  Epoch {epoch:04d} | loss={loss.item():.5f}")
            
            # Compute baseline hazard for this cause
            with torch.no_grad():
                model_k.eval()
                r_train_k = model_k(X_tensor).detach().cpu().numpy()
            
            etimes_k, H0_k = _breslow_baseline_hazard(r_train_k, durations, events_k)
            
            self.models_.append(model_k)
            self.baseline_cum_hazards_.append(H0_k)
            event_times_per_cause.append(etimes_k)
        
        # Store all unique event times across all causes
        all_event_times = np.concatenate(event_times_per_cause)
        self.event_times_ = np.unique(all_event_times)
        
        return self
    
    def predict_cause_specific_risk(
        self, 
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame], 
        cause: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict cause-specific hazards (log-risk scores).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        
        cause : int or None
            If int in [1, K], return risk for that cause only.
            If None, return all K cause-specific risks.
        
        Returns
        -------
        risks : np.ndarray
            Shape (n_samples,) if cause specified.
            Shape (n_samples, K) if cause=None.
        """
        check_is_fitted(self, "models_")
        
        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)
        
        if self.backbone.lower() == "tabicl":
            embeddings = self._extract_tabicl_embeddings(X)
        else:
            embeddings = self._standardize_inputs(np.asarray(X))
        
        X_tensor = _to_tensor(embeddings, self.device_)
        
        risks = []
        for model_k in self.models_:
            with torch.no_grad():
                risk_k = model_k(X_tensor).detach().cpu().numpy()
            risks.append(risk_k)
        
        risks = np.stack(risks, axis=1)  # (n_samples, K)
        
        if cause is not None:
            if not (1 <= cause <= self.n_event_types):
                raise ValueError(f"cause must be in [1, {self.n_event_types}], got {cause}")
            return risks[:, cause - 1]
        
        return risks
    
    def score(
        self, 
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame], 
        y: Tuple[Iterable, Iterable],
        cause: Optional[int] = None,
    ) -> float:
        """
        Compute cause-specific C-index.
        
        Parameters
        ----------
        X : array-like
            Input features.
        
        y : tuple of (durations, event_types)
            True survival outcomes.
        
        cause : int or None
            If int, return C-index for that specific cause.
            If None, return weighted average across all causes.
        
        Returns
        -------
        c_index : float
            Concordance index (0.5 = random, 1.0 = perfect).
        """
        durations, event_types = y
        durations = np.asarray(durations, dtype=float)
        event_types = np.asarray(event_types, dtype=int)
        
        if cause is not None:
            # C-index for specific cause
            events_k = (event_types == cause).astype(float)
            if events_k.sum() == 0:
                return np.nan  # No events of this type
            
            risk_k = self.predict_cause_specific_risk(X, cause=cause)
            return _concordance_index(durations, events_k, risk_k)
        else:
            # Weighted average across all causes
            c_indices = []
            weights = []
            
            for k in range(1, self.n_event_types + 1):
                events_k = (event_types == k).astype(float)
                n_events_k = events_k.sum()
                
                if n_events_k > 0:
                    risk_k = self.predict_cause_specific_risk(X, cause=k)
                    c_k = _concordance_index(durations, events_k, risk_k)
                    c_indices.append(c_k)
                    weights.append(n_events_k)
            
            if len(c_indices) == 0:
                return np.nan
            
            # Weighted average by number of events
            weights = np.array(weights) / sum(weights)
            return np.average(c_indices, weights=weights)


# Add sklearn tags method for compatibility
if not OLD_SKLEARN:
    def _sklearn_tags_method(self):
        """Sklearn tags for compatibility with sklearn >= 1.6."""
        tags = super(TabICLCompetingRisks, self).__sklearn_tags__()
        tags.non_deterministic = True
        return tags
    
    TabICLCompetingRisks.__sklearn_tags__ = _sklearn_tags_method

