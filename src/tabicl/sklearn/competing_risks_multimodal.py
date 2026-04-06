from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path

import math
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
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
from ..train.multi_armed_bandit import LinUCBTopK, LinUCBConfig
from ..train.gait_encoder import CoxHeadWrapper


class TabICLCompetingRisksMultiModal(BaseEstimator):    
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
        multi_cox_heads: bool = False
    ):
        super().__init__()
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
        self.multi_cox_heads = multi_cox_heads
    
    def _more_tags(self):
        """Mark estimator as non-deterministic."""
        return dict(non_deterministic=True)
    
    # Reuse methods from TabICLSurvivor by copying implementation
    def _load_tabicl_model(self):
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
    
    @staticmethod
    def compute_rewards(attn, grad_z_gait):
        """
            We scale by ||grad_z_gait||_2 per subject.
            Returns reward per window (B, 128)
        """
        g_norm = torch.linalg.vector_norm(grad_z_gait, ord=2, keepdim=True)  # (1,)
        rewards = attn * g_norm  # (128,)

        # z-score per subject to stabilize bandit updates
        mu = rewards.mean()
        sd = rewards.std() + 1e-6
        rewards = (rewards - mu) / sd
        return rewards  # (128,)

    @staticmethod
    def collect_risk_scores(model_k, dataloader, cause, device, use_amp=True):
        # Preallocate on CPU
        N = len(dataloader.dataset)
        r_all = torch.empty(N, dtype=torch.float32) if cause is not None else torch.empty((N, 7), dtype=torch.float32)
        t_all = torch.empty(N, dtype=torch.float32)
        e_all = torch.empty(N, dtype=torch.float32)
        idx_winds_all = torch.empty((N, 128), dtype=torch.long)

        with torch.no_grad():
            for x_ts, x_tab, mask, (y_times, y_events), idx in dataloader:
                x_ts, x_tab, mask = x_ts.to(device), x_tab.to(device), mask.to(device)

                if isinstance(idx, list):
                    idx, idx_winds = zip(*idx)
                    idx_winds = torch.stack(idx_winds)
                idx = torch.Tensor(idx).long()
    
                # Competing risks for cause k, others are censored
                events_k = (y_events == cause).float() if cause is not None else y_events.float()

                # Forward (no grad)
                with autocast(device_type=device, enabled=use_amp):
                    risks, _, _ = model_k(x_ts, x_tab, mask)
                    # risks, _ = model_k(x_ts, x_tab, mask)

                r_all[idx] = risks.detach().float().cpu()
                t_all[idx] = y_times
                e_all[idx] = events_k

                # cache window indexes per subject
                idx_winds_all[idx] = idx_winds

        return r_all, t_all, e_all, idx_winds_all


    def train_epoch(self, model_k, train_loader, opt_k, device, k=None, use_amp=True):
        model_k.train()
        N = len(train_loader.dataset)

        # cache attention grad and z_gait grad for reward calculation
        try:
            z_dim = model_k.encoder_ts.out_dim if not isinstance(model_k, nn.DataParallel) else model_k.module.encoder_ts.out_dim
        except:
            z_dim = 128
        idx_winds_all = torch.empty((N, 128), dtype=torch.long)
        grad_attn_all = torch.empty((N, 128), dtype=torch.float32, device='cpu')
        grad_zg_all = torch.empty((N, z_dim), dtype=torch.float32, device='cpu')

        loss_value = 0.0

        for x_ts, x_tab, mask, (y_times, y_events), idx in train_loader:
            x_ts, x_tab, mask = x_ts.to(device), x_tab.to(device), mask.to(device)
            y_events, y_times = y_events.to(device), y_times.to(device)

            if isinstance(idx, list):
                idx, idx_winds = zip(*idx)
                idx_winds = torch.stack(idx_winds)
            idx = torch.Tensor(idx).long()

            # Competing risks for cause k, others are censored
            events_k = y_events.float()

            # Forward
            with autocast(device_type=device, enabled=False, dtype=torch.float16):
                risks, z_gait, scores = model_k(x_ts, x_tab, mask)

            # risks.retain_grad()
            # z_gait.retain_grad()
            # scores.retain_grad()

            # Calculate Cox loss in fp32
            # loss = 0.0
            loss = torch.zeros((), device=risks.device, dtype=torch.float32)

            for i in range(risks.shape[1]):
                e_k = (events_k == i + 1).float()
                loss = loss + _coxph_breslow_loss(risks[:, i].float(), y_times, e_k, None)
            
            # g_scores, g_z = torch.autograd.grad(
            #     outputs=loss,
            #     inputs=(scores, z_gait),
            #     retain_graph=True,
            #     allow_unused=True
            # )

            # if g_scores is None:
            #     g_scores = torch.zeros_like(scores)
            # if g_z is None:
            #     g_z = torch.zeros_like(z_gait)

            # # Capture grads
            # g_scores_cpu = g_scores.detach().float().cpu()
            # g_z_cpu = g_z.detach().float().cpu()

            # loss_value = loss.item()

            opt_k.zero_grad(set_to_none=True)
            loss.backward()
            opt_k.step()

            # # Cache reward signals on cpu
            # grad_zg_all[idx] = g_z_cpu
            # grad_attn_all[idx] = g_scores_cpu
            # idx_winds_all[idx] = idx_winds

            # Update loss value for monitoring
            loss_value += loss.item()

        return loss_value/len(train_loader), grad_zg_all, grad_attn_all, idx_winds_all

    def train_epoch_two_pass_exact(self, model_k, train_loader, opt_k, device, k=None, use_amp=True):
        model_k.train()
        opt_k.zero_grad(set_to_none=True)

        # PASS A: run through dataset, collect risk scores (NO GRAD)
        r_all, t_all, e_all, idx_winds_all = self.collect_risk_scores(model_k, train_loader, cause=k, device=device, use_amp=True)

        with torch.enable_grad():
            r_all = r_all.to(device).detach().clone().requires_grad_(True)
            t_all = t_all.to(device)
            e_all = e_all.to(device)

            # Calculate loss
            if k is not None:
                loss = _coxph_breslow_loss(r_all, t_all, e_all, None)
            else:
                K = r_all.shape[1]
                loss = 0.0
                for i in range(K):
                    e_k = (e_all == i + 1).float()
                    loss += _coxph_breslow_loss(r_all[:, i], t_all, e_k, None)

            loss_value = loss.item()

            # Exact gradient with respect to risks (per samplegradient dL/dr for the whole dataset)
            grad_map = torch.autograd.grad(loss, r_all, retain_graph=False)[0].detach()

            # cache attention grad and z_gait grad for reward calculation
            N = r_all.shape[0]
            z_dim = model_k.encoder_ts.out_dim if not isinstance(model_k, nn.DataParallel) else model_k.module.encoder_ts.out_dim
            grad_attn_all = torch.empty((N, 128), dtype=torch.float32, device='cpu')
            grad_zg_all = torch.empty((N, z_dim), dtype=torch.float32, device='cpu')

            # PASS B: re-forward bachwise and backprop with custom gradient g = dL/dr
            for x_ts, x_tab, mask, (y_times, y_events), idx in tqdm(train_loader):
                x_ts, x_tab = x_ts.to(device), x_tab.to(device)
                mask = mask.to(device)

                if isinstance(idx, list):
                    idx, idx_winds = zip(*idx)
                    idx_winds = torch.stack(idx_winds)
                idx = torch.Tensor(idx).long()

                with autocast(device_type=device, enabled=use_amp):
                    risks, z_gait, attn = model_k(x_ts, x_tab, mask)
                    # risks, z_gait = model_k(x_ts, x_tab, mask)

                # retain to get grad wrt z_gait (z_gait is non-leaf)
                z_gait.retain_grad()

                # take the corresponding gradient entries for this batch  (f32)
                g = grad_map[idx].to(risks.device)
                risks = risks.float()
                g = g.view_as(risks).to(risks.dtype)

                # backpropagate custom per-sample gradient through the model
                risks.backward(gradient=g)

                # cache reward signals on cpu
                grad_zg_all[idx] = z_gait.grad.detach().cpu().float()
                grad_attn_all[idx] = attn.detach().cpu().float()

        # Single optimizer step for the full-dataset gradient
        opt_k.step()

        return loss_value, grad_zg_all, grad_attn_all, idx_winds_all

    def fit(
        self, train_dataloader, val_dataloader, 
    ) -> TabICLCompetingRisksMultiModal:
        X = train_dataloader.dataset.x_tab.float()
        X_val = val_dataloader.dataset.x_tab.float()

        # Validate data
        if OLD_SKLEARN:
            X = self._validate_data(X, dtype=None, cast_to_ndarray=False, reset=True)
        else:
            X = self._validate_data(X, dtype=None, skip_check_array=True, reset=True)
                
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
        # if X_val is not None and y_val is not None:
        if X_val is not None:
            if self.backbone.lower() == "tabicl":
                embeddings_val = self._extract_tabicl_embeddings(X_val)
            else:
                embeddings_val = self._standardize_inputs(np.asarray(X_val))
        else:
            embeddings_val = None
        
        # Update the embeddings in the datasets
        train_dataloader.dataset.x_tabicl = torch.from_numpy(embeddings)
        val_dataloader.dataset.x_tabicl = torch.from_numpy(embeddings_val)

        # Train K cause-specific models and bandits
        self.models_, self.bandits_ = [], []
        self.baseline_cum_hazards_ = []
        self.event_times_per_cause_ = []  # Store per-cause event times
        device = self.device

        if not self.multi_cox_heads:
            for k in range(1, self.n_event_types + 1):        
                # Train Cox model for this cause
                best_val = math.inf
                best_state = None
                no_improve = 0

                model_k = CoxHeadWrapper().to(device)
                opt_k = torch.optim.AdamW(model_k.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                bandit_k = LinUCBTopK(LinUCBConfig())      

                ##### Train function was here #####

                for epoch in tqdm(range(self.epochs)):
                    bandit_k.run_selection_for_epoch(train_dataloader.dataset)
                    loss, grad_zg_all, grad_attn_all, idx_winds_all = self.train_epoch_two_pass_exact(model_k, train_dataloader, opt_k, device, k=k, use_amp=True)
                    
                    # Bandit update using rewards
                    with torch.no_grad():
                        for sid, (idx_winds, gz, a) in enumerate(zip(idx_winds_all, grad_zg_all, grad_attn_all)):
                            rewards = self.compute_rewards(a.to(device), gz.to(device))  # (128,)
                            idx_winds_np = idx_winds.numpy()  # .astype(np.int64)
                            E_sel = train_dataloader.dataset.get_gait_embeddings_by_sid(sid, idx_winds_np)  # (128, 1024)
                            r_np = rewards.cpu().numpy().astype(np.float32)  # (128,)
                            bandit_k.update(E_sel, r_np)

                    # Validation
                    if embeddings_val is not None:
                        model_k.eval()
                        bandit_k.run_selection_for_epoch(val_dataloader.dataset)
                        rv_all, tv_all, ev_all, _ = self.collect_risk_scores(model_k, dataloader=val_dataloader, cause=k, device=device)            
                        lv = _coxph_breslow_loss(rv_all,  tv_all, ev_all, None).item()
                        
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
                    
                    if self.verbose and epoch % 1 == 0:
                        print(f"  Epoch {epoch:04d} | loss={loss:.5f}")
                
                # Compute baseline hazard for this cause
                torch.save(model_k.state_dict(), f'survival_best_model_cause_{k}_tabicl_multi_mab_resnet.mdl')
                bandit_k.run_selection_for_epoch(train_dataloader.dataset)
                model_k.eval()
                r_train_k, durations, events, _ = self.collect_risk_scores(model_k, dataloader=train_dataloader, cause=k, device=device)            
                etimes_k, H0_k = _breslow_baseline_hazard(r_train_k, durations, events)

                # Cache model and parameters            
                self.models_.append(model_k)
                self.bandits_.append(bandit_k)
                self.baseline_cum_hazards_.append(H0_k)
                self.event_times_per_cause_.append(etimes_k)
            
            # Store all unique event times across all causes
            all_event_times = np.concatenate(self.event_times_per_cause_)
            self.event_times_ = np.unique(all_event_times)

        else:
            load_pretrained_cox_model = False
            model_name = 'survival_best_model_multi_cox_heads_tabicl_mab_v2_resnet.mdl'
            if load_pretrained_cox_model:
                print('Skipping training Cox Model, loading a trained model from: \n', model_name)
                model_dict = torch.load(model_name, map_location='cuda')
                model_k = CoxHeadWrapper(multi_cox_heads=True).to(device)
                model_k.load_state_dict(model_dict)

                self.models_.append(model_k)
                bandit_k = LinUCBTopK(LinUCBConfig())      
                self.bandits_.append(bandit_k)
                return self
            
            # Train Cox model for all K causes
            K = self.n_event_types
            best_val = -math.inf
            print('Monitoring Ctd Metric and Monitoring MAXIMAL change (best_val) is minimum Inf !!!!!')
            best_state = None
            no_improve = 0

            model_k = CoxHeadWrapper(multi_cox_heads=self.multi_cox_heads)  # .to(device)

            # Multi-GPU handling
            # model_k = nn.DataParallel(model_k, device_ids=[0, 1])
            model_k.to(device, dtype=torch.float)

            opt_k = torch.optim.AdamW(model_k.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            bandit_k = LinUCBTopK(LinUCBConfig())      

            for epoch in tqdm(range(self.epochs)):
                bandit_k.run_selection_for_epoch(train_dataloader.dataset)
                # train_loss, grad_zg_all, grad_attn_all, idx_winds_all = self.train_epoch_two_pass_exact(model_k, train_dataloader, opt_k, device, use_amp=True)
                train_loss, grad_zg_all, grad_attn_all, idx_winds_all = self.train_epoch(model_k, train_dataloader, opt_k, device, use_amp=True)
                
                # # Bandit update using rewards
                # with torch.no_grad():
                #     for sid, (idx_winds, gz, a) in enumerate(zip(idx_winds_all, grad_zg_all, grad_attn_all)):
                #         rewards = self.compute_rewards(a.to(device), gz.to(device))  # (128,)
                #         idx_winds_np = idx_winds.numpy()  # .astype(np.int64)
                #         E_sel = train_dataloader.dataset.get_gait_embeddings_by_sid(sid, idx_winds_np)  # (128, 1024)
                #         r_np = rewards.cpu().numpy().astype(np.float32)  # (128,)
                #         bandit_k.update(E_sel, r_np)

                # Validation
                if embeddings_val is not None:
                    model_k.eval()
                    bandit_k.run_selection_for_epoch(val_dataloader.dataset)
                    rv_all, tv_all, ev_all, _ = self.collect_risk_scores(model_k, dataloader=val_dataloader, cause=None, device=device)
                    
                    # Calculate validation loss
                    loss = 0.0
                    for i in range(K):
                        e_k = (ev_all == i + 1).float()
                        loss += _coxph_breslow_loss(rv_all[:, i], tv_all, e_k, None)
                    lv = loss.item()

                    c_indices, weights = [], []
                    ev_all = ev_all.numpy().astype(int)
                    tv_all =tv_all.numpy().astype(float)
                    
                    for k in range(1, self.n_event_types + 1):
                        events_k = (ev_all == k).astype(float)
                        n_events_k = events_k.sum()
                        
                        if n_events_k > 0:
                            risk_k = rv_all[:, k - 1]
                            c_k = _concordance_index(tv_all, events_k, risk_k)
                            c_indices.append(c_k)
                            weights.append(n_events_k)
                    
                    # Weighted average by number of events
                    weights = np.array(weights) / sum(weights)
                    c_tabicl_multi_val = np.average(c_indices, weights=weights)
                    
                    # if lv + 1e-9 < best_val:
                    #     best_val = lv
                    #     best_state = {k: v.detach().cpu().clone() for k, v in model_k.state_dict().items()}
                    #     no_improve = 0
                    if c_tabicl_multi_val + 1e-9 > best_val:
                        best_val = c_tabicl_multi_val
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
                
                if self.verbose and epoch % 1 == 0:
                    print(f"  Epoch {epoch:04d} | train_loss={train_loss:.5f}, val_loss={lv:.5f}, c-index-avg={c_tabicl_multi_val:.5f}")
            
            # Compute baseline hazard for this cause
            torch.save(model_k.state_dict(), f'survival_best_model_multi_cox_heads_tabicl_mab_v2_resnet.mdl')
            bandit_k.run_selection_for_epoch(train_dataloader.dataset)
            model_k.eval()
            r_train_k, durations, events, _ = self.collect_risk_scores(model_k, dataloader=train_dataloader, cause=None, device=device)
            for i in range(K):
                e_k = (events == i + 1).float()
                etimes_k, H0_k = _breslow_baseline_hazard(r_train_k[:, i], durations, e_k)

                # Cache model and parameters            
                self.baseline_cum_hazards_.append(H0_k)
                self.event_times_per_cause_.append(etimes_k)

            self.models_.append(model_k)
            self.bandits_.append(bandit_k)
        
            # Store all unique event times across all causes
            all_event_times = np.concatenate(self.event_times_per_cause_)
            self.event_times_ = np.unique(all_event_times)
        
        return self
    
    def predict_cause_specific_risk(self, dataloader, cause=None):
        check_is_fitted(self, "models_")
        X = dataloader.dataset.x_tab.float()

        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)
        
        if self.backbone.lower() == "tabicl":
            embeddings = self._extract_tabicl_embeddings(X)
        else:
            embeddings = self._standardize_inputs(np.asarray(X))
        
        # Update the embeddings in the datasets
        dataloader.dataset.x_tabicl = torch.from_numpy(embeddings)
        
        all_risks = []

        for i, model_k in enumerate(self.models_):
            model_k.eval()

            # Run selection
            print(f"Predicting Cause {i}")
            self.bandits_[i].run_selection_for_epoch(dataloader.dataset)

            risk_chunks = []
            with torch.no_grad():
                for x_ts, x_tab, mask, (_, _), _ in tqdm(dataloader):
                    x_ts, x_tab = x_ts.to(self.device), x_tab.to(self.device)
                    mask = mask.to(self.device)

                    with autocast(device_type=self.device, enabled=True):
                        risks, _, _ = model_k(x_ts, x_tab, mask)

                    risk_chunks.append(risks.detach().cpu())
            all_risks.append(torch.cat(risk_chunks).numpy())

        all_risks = np.stack(all_risks, axis=1)  # (n_samples, K)

        # Handle case of 1 model for all causes
        if all_risks.ndim == 3 and all_risks.shape[1] == 1:
            all_risks = all_risks.squeeze(1)

        if cause is not None:
            if not (1 <= cause <= self.n_event_types):
                raise ValueError(f"cause must be in [1, {self.n_event_types}], got {cause}")
            return all_risks[:, cause - 1]
        
        return all_risks
    
    def predict_cumulative_incidence(
        self,
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        times: Optional[Iterable[float]] = None,
        return_array: bool = False,
        cause: Optional[int] = None,
    ) -> Union[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict[int, List[Tuple[np.ndarray, np.ndarray]]], Dict[int, np.ndarray]]:
        check_is_fitted(self, ["models_", "baseline_cum_hazards_", "event_times_per_cause_", "event_times_"])
        
        if OLD_SKLEARN:
            X = self._validate_data(X, reset=False, dtype=None, cast_to_ndarray=False)
        else:
            X = self._validate_data(X, reset=False, dtype=None, skip_check_array=True)
        
        # Get embeddings
        if self.backbone.lower() == "tabicl":
            embeddings = self._extract_tabicl_embeddings(X)
        else:
            embeddings = self._standardize_inputs(np.asarray(X))
        
        X_tensor = _to_tensor(embeddings, self.device_)
        n_samples = embeddings.shape[0]
        
        # Determine time grid
        if times is None:
            times = self.event_times_
        times = np.asarray(times, dtype=float)
        times = np.sort(times)
        n_times = len(times)
        
        # Get risk scores for all causes
        all_risks = []
        for i, model_k in enumerate(self.models_):
            model_k.eval()

            # print(f"Predicting Cause {i}")
            # self.bandits_[i].run_selection_for_epoch(dataloader.dataset)

            with torch.no_grad():
                risk_k = model_k(X_tensor).detach().cpu().numpy()
            all_risks.append(risk_k)  # Each is shape (n_samples,)
        
        # Compute CIF for each cause
        def compute_cif_for_cause(cause_idx):
            """Compute CIF for cause_idx (0-indexed, so cause_idx=0 means cause 1)."""
            cif_samples = []
            for i in range(n_samples):
                # Compute cause-specific cumulative hazards for all causes at each time
                H_all_causes = np.zeros((self.n_event_types, n_times))
                
                for k_idx in range(self.n_event_types):
                    risk_k_i = all_risks[k_idx][i]
                    H0_k = self.baseline_cum_hazards_[k_idx]
                    event_times_k = self.event_times_per_cause_[k_idx]
                    
                    # Interpolate baseline cumulative hazard to target times
                    H0_k_at_times = np.interp(
                        times, 
                        event_times_k,
                        H0_k,
                        left=0,
                        right=H0_k[-1] if len(H0_k) > 0 else 0
                    )
                    
                    # Individual cumulative hazard: H_k(t) = H0_k(t) * exp(risk_k)
                    H_all_causes[k_idx, :] = H0_k_at_times * np.exp(risk_k_i)
                
                # Overall survival: S(t) = exp(-Σ_k H_k(t))
                H_total = H_all_causes.sum(axis=0)  # Sum over causes
                S_t = np.exp(-H_total)
                
                # CIF for this cause: CIF_k(t) = ∫_0^t S(u-) dH_k(u)
                H_k = H_all_causes[cause_idx, :]
                
                # Numerical integration
                # Prepend 0 to start integration from 0
                H_k_with_zero = np.concatenate([[0], H_k])
                S_t_with_zero = np.concatenate([[1.0], S_t])
                
                # CIF increment at each time: S(t-) * dH_k
                dH_k = np.diff(H_k_with_zero)
                # Use S at left endpoint (S(t-))
                S_left = S_t_with_zero[:-1]
                
                cif_increments = S_left * dH_k
                cif = np.cumsum(cif_increments)
                
                cif_samples.append(cif)
            
            return times, np.array(cif_samples)  # (n_samples, n_times)
        
        # Compute for requested causes
        if cause is not None:
            if not (1 <= cause <= self.n_event_types):
                raise ValueError(f"cause must be in [1, {self.n_event_types}], got {cause}")
            
            times_out, cif_array = compute_cif_for_cause(cause - 1)
            
            if return_array:
                return cif_array
            else:
                # Return list of (times, CIF) tuples
                return [(times_out, cif_array[i, :]) for i in range(n_samples)]
        else:
            # Return all causes
            cif_dict = {}
            for k in range(1, self.n_event_types + 1):
                times_out, cif_array = compute_cif_for_cause(k - 1)
                
                if return_array:
                    cif_dict[k] = cif_array
                else:
                    cif_dict[k] = [(times_out, cif_array[i, :]) for i in range(n_samples)]
            
            return cif_dict


    def score(self, dataloader, y: Tuple[Iterable, Iterable], cause: Optional[int] = None):
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
            
            risk_k = self.predict_cause_specific_risk(dataloader, cause=cause)
            return _concordance_index(durations, events_k, risk_k)
        else:
            # Weighted average across all causes
            c_indices = []
            weights = []
            
            for k in range(1, self.n_event_types + 1):
                events_k = (event_types == k).astype(float)
                n_events_k = events_k.sum()
                
                if n_events_k > 0:
                    risk_k = self.predict_cause_specific_risk(dataloader, cause=k)
                    c_k = _concordance_index(durations, events_k, risk_k)
                    c_indices.append(c_k)
                    weights.append(n_events_k)
            
            if len(c_indices) == 0:
                return np.nan
            
            # Weighted average by number of events
            weights = np.array(weights) / sum(weights)
            return np.average(c_indices, weights=weights)
