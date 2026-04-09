# TabPFN Competing Risks — Setup & Usage

This document describes how to install TabPFN, pre-download its weights for
offline use, and run the `TabPFNCompetingRisks` model in a closed (no-internet)
environment.

---

## 1. What is TabPFNCompetingRisks?

`TabPFNCompetingRisks` (in `src/tabicl/sklearn/tabpfn_competing_risks.py`) uses
the pretrained **TabPFN v2** transformer as a frozen feature encoder in place of
TabICL.  The workflow mirrors `TabICLCompetingRisks`:

```
X (tabular) ──► TabPFN (frozen) ──► embedding ──► _MLPRisk × K ──► risk scores
```

Key difference from TabICL: TabPFN is an **in-context learner**, so each embedding
depends on a training-set context.  To avoid leakage when embedding training samples,
the model uses **out-of-fold (OOF)** extraction automatically during `fit()`.

---

## 2. Installation

### 2a. On an internet-connected machine (prepare for offline transfer)

```bash
# 1. Install tabpfn from PyPI
pip install tabpfn

# 2. Trigger the weight download by running a quick dummy fit.
#    This downloads the TabPFN-2.6 checkpoint from HuggingFace into your cache.
python - <<'EOF'
import numpy as np
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(device="cpu")
clf.fit(np.random.randn(20, 5), np.array([0]*10 + [1]*10))
print("Download complete. Embedding dim:", clf.get_embeddings(np.random.randn(3, 5)).shape)
EOF
```

The checkpoint is saved in the HuggingFace cache:

| OS      | Default cache path                                    |
|---------|-------------------------------------------------------|
| Windows | `C:\Users\<user>\.cache\huggingface\hub\`             |
| Linux   | `~/.cache/huggingface/hub/`                           |
| macOS   | `~/.cache/huggingface/hub/`                           |

You can override this with the environment variable `HF_HOME`:

```bash
# Optional: use a custom cache directory (easier to copy)
export HF_HOME=/path/to/my_hf_cache
python -c "from tabpfn import TabPFNClassifier; TabPFNClassifier().fit([[0]*5],[0])"
```

### 2b. Transfer to the closed environment

1. Copy the HuggingFace cache directory to the closed machine:
   ```
   # Example (Windows → Windows via USB / shared drive)
   robocopy C:\Users\<you>\.cache\huggingface  \\closed-machine\share\hf_cache /E
   ```

2. On the closed machine, set the cache path before running anything:
   ```bash
   set HF_HOME=\\closed-machine\share\hf_cache       # Windows CMD
   # OR
   $env:HF_HOME = "\\closed-machine\share\hf_cache"  # PowerShell
   # OR
   export HF_HOME=/path/to/hf_cache                  # Linux/macOS
   ```

3. Tell HuggingFace to use only local files (no HTTP requests):
   ```bash
   set HF_HUB_OFFLINE=1       # Windows CMD
   $env:HF_HUB_OFFLINE = "1"  # PowerShell
   export HF_HUB_OFFLINE=1    # Linux/macOS
   ```

### 2c. Install tabpfn on the closed machine

If pip can reach an internal mirror or you have a wheel file:

```bash
# From a pre-downloaded wheel
pip install tabpfn-2.x.x-py3-none-any.whl

# From an internal mirror
pip install tabpfn --index-url http://your-internal-pypi-mirror/simple/
```

To download the wheel on the internet machine:
```bash
pip download tabpfn -d ./tabpfn_wheels/
# Transfer ./tabpfn_wheels/ to closed machine, then:
pip install --no-index --find-links=./tabpfn_wheels/ tabpfn
```

---

## 3. Verify the setup (closed machine)

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"path\to\hf_cache"   # adjust

import numpy as np
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
clf.fit(np.random.randn(20, 5), np.array([0]*10 + [1]*10))
emb = clf.get_embeddings(np.random.randn(5, 5))
print("OK — embedding shape:", emb.shape)   # e.g. (5, 512)
```

If this runs without an HTTP error the environment is correctly configured.

---

## 4. Usage — TabPFNCompetingRisks

```python
import os
# ---- set before any huggingface import ----
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"path\to\hf_cache"

import numpy as np
from tabicl.sklearn.tabpfn_competing_risks import TabPFNCompetingRisks

# --- data (replace with your WHS arrays) ---
X_train, X_val, X_test = ...   # shape (n, n_features), dtype float32
durations_train, events_train  = ...   # durations: float, events: int {0,1,...,K}
durations_val,   events_val    = ...
durations_test,  events_test   = ...

# --- model ---
model = TabPFNCompetingRisks(
    n_event_types      = 7,          # number of competing outcomes
    n_oof_splits       = 5,          # OOF folds for train embedding (no leakage)
    max_context_samples= 8000,       # subsample context if n_train > TabPFN limit
    ignore_pretraining_limits = True,
    hidden             = 64,         # Cox head hidden dim (None = linear)
    dropout            = 0.1,
    lr                 = 1e-3,
    weight_decay       = 1e-4,
    epochs             = 200,
    patience           = 20,
    device             = "cuda",     # or "cpu"
    random_state       = 42,
    verbose            = True,
)

model.fit(
    X_train, (durations_train, events_train),
    X_val=X_val, y_val=(durations_val, events_val),
)

# --- evaluate ---
c_index = model.score(X_test, (durations_test, events_test))
print(f"Weighted avg C-index: {c_index:.4f}")

# Per-cause C-index
for k in range(1, 8):
    c_k = model.score(X_test, (durations_test, events_test), cause=k)
    print(f"  Cause {k}: C = {c_k:.4f}")

# Cause-specific risk scores  (n_test, K)
risks = model.predict_cause_specific_risk(X_test)

# Cumulative incidence functions  dict{cause → (n_test, n_times)}
cifs = model.predict_cumulative_incidence(X_test, return_array=True)
```

---

## 5. Important parameters

| Parameter               | Default | Notes                                                        |
|-------------------------|---------|--------------------------------------------------------------|
| `n_oof_splits`          | 5       | More folds = less leakage, slower fit.                       |
| `max_context_samples`   | None    | Set to ≤ 10 000 if TabPFN raises a size warning.             |
| `ignore_pretraining_limits` | True | Allows datasets larger than TabPFN's training distribution.  |
| `hidden`                | None    | None = linear Cox head; int = one hidden ReLU layer.         |
| `patience`              | 20      | Requires `X_val` / `y_val` to activate early stopping.      |
| `device`                | None    | Auto-detects CUDA. Use `"cpu"` on CPU-only servers.          |

---

## 6. Fit time expectations

| Phase                     | Approx time (n=17 000, 30 features, 5 folds, CPU) |
|---------------------------|----------------------------------------------------|
| Full-context TabPFN fit   | ~30 s                                              |
| OOF embedding (5 folds)   | ~2.5 min (5 × 30 s)                               |
| Val embedding             | ~30 s                                              |
| Cox heads × 7 causes      | ~2 min                                             |
| **Total (CPU)**           | **~5 – 6 min per CV fold**                         |

GPU will be significantly faster for the TabPFN forward passes.

---

## 7. Troubleshooting

| Error                                  | Fix                                                       |
|----------------------------------------|-----------------------------------------------------------|
| `ModuleNotFoundError: tabpfn`          | `pip install tabpfn` (or install from wheel offline)      |
| `get_embeddings() not found`           | Upgrade to tabpfn >= 2.0                                  |
| HTTP / network timeout on closed machine | Set `HF_HUB_OFFLINE=1` and `HF_HOME` before import    |
| `UserWarning: n_samples exceeds …`    | Set `max_context_samples=8000` or `ignore_pretraining_limits=True` |
| CUDA out of memory                     | Use `device="cpu"` or reduce `max_context_samples`        |
