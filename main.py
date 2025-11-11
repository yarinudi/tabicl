import numpy as np
import xgboost as xgb

from src.tabicl.sklearn.survivor import TabICLSurvivor


# ##### XGBoost Survival Helpers
def build_cause_specific_labels(times, events):
    """Build cause-specific labels for XGBoost survival model."""
    _, K = times.shape
    time_k, event_k = [], []
    
    for k in range(K):
        time_k.append(times[:, k])
        event_k.append((events[:, k]).astype(np.int64))
    
    return time_k, event_k


def train_xgb_cox_survival(x, times, events, val=None, params=None, num_boost_round=1000):
    """Train XGBoost Cox survival model with cause-specific labels."""
    if params is None:
        params = {
            "seed": 42,
            "objective": 'survival:cox',
            "eval_metric": 'cox-nloglik',
            "tree_method": 'hist',
            "learning_rate": 0.05,
            "max_depth": 1,  # Decision stumps
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    
    time_k, event_k = build_cause_specific_labels(times, events)
    if val:
        x_val, (times_val, events_val) = val
        time_k_val, event_k_val = build_cause_specific_labels(times_val, events_val)
    
    models = []
    watchlist = None
    for k in range(len(time_k)):
        # Build signed labels: positive for events, negative for censored
        y_signed = np.where(event_k[k] == 1, time_k[k], -time_k[k]).astype(np.float32)
        dtrain = xgb.DMatrix(x, label=y_signed)
        
        if val:
            y_signed_val = np.where(event_k_val[k] == 1, time_k_val[k], -time_k_val[k]).astype(np.float32)
            dval = xgb.DMatrix(x_val, label=y_signed_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        bst = xgb.train(
            params, 
            dtrain, 
            evals=watchlist, 
            early_stopping_rounds=50, 
            verbose_eval=False,
            num_boost_round=num_boost_round
        )
        models.append(bst)
    
    return models


def predict_xgb_cox(models, x):
    """Predict risk scores from XGBoost Cox models."""
    # Use first model (single event type in our case)
    dtest = xgb.DMatrix(x)
    return models[0].predict(dtest)


if __name__ == "__main__":
    # Generate synthetic survival data with true signal
    np.random.seed(42)
    n_train, n_test = 400, 100
    n_features = 10
    
    # Create features with different types
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)
    
    # Define true coefficients (some features are prognostic, others are noise)
    # Features 0-2: strong prognostic factors
    # Features 3-5: moderate prognostic factors  
    # Features 6-9: noise (no effect)
    true_beta = np.array([
        0.8,   # Feature 0: strong positive effect (higher = shorter survival)
        -0.7,  # Feature 1: strong negative effect (higher = longer survival)
        0.5,   # Feature 2: moderate positive effect
        -0.3,  # Feature 3: weak negative effect
        0.2,   # Feature 4: weak positive effect
        -0.2,  # Feature 5: weak negative effect
        0.0,   # Features 6-9: no effect (noise)
        0.0,
        0.0,
        0.0
    ])
    
    def generate_survival_data(X, beta, base_scale=100, censoring_rate=0.3):
        """
        Generate survival times using Cox proportional hazards model:
        λ(t|X) = λ₀(t) * exp(β'X)
        
        For exponential baseline: T ~ Exponential(λ₀ * exp(β'X))
        """
        n = X.shape[0]
        
        # Compute log-hazard (linear predictor)
        log_hazard = X @ beta
        
        # Generate survival times from exponential distribution
        # Scale inversely proportional to hazard: higher hazard = shorter survival
        hazard_ratios = np.exp(log_hazard)
        scales = base_scale / hazard_ratios
        
        # Generate true survival times
        true_times = np.random.exponential(scale=scales)
        
        # Generate censoring times independently
        censoring_times = np.random.exponential(scale=base_scale / (1 - censoring_rate), size=n)
        
        # Observed time is minimum of true time and censoring time
        observed_times = np.minimum(true_times, censoring_times)
        
        # Event indicator: 1 if event observed, 0 if censored
        events = (true_times <= censoring_times).astype(float)
        
        return observed_times, events
    
    # Generate training data with true signal
    t_train, e_train = generate_survival_data(X_train, true_beta, base_scale=100, censoring_rate=0.3)
    
    # Generate test data with same true coefficients
    t_test, e_test = generate_survival_data(X_test, true_beta, base_scale=100, censoring_rate=0.3)
    
    print(f"Training: {n_train} samples, {e_train.sum():.0f} events ({100*e_train.mean():.1f}% observed)")
    print(f"Test: {n_test} samples, {e_test.sum():.0f} events ({100*e_test.mean():.1f}% observed)")
    print(f"True coefficients (first 6): {true_beta[:6]}")
    print()

    # Model with TabICL backbone (uses pretrained TabICL for feature extraction)
    model = TabICLSurvivor(
        backbone="tabicl",
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
        hidden=128,
        epochs=100,
        patience=15,
        verbose=False  # Set to True to see training progress
    )

    print("Fitting survival model...")
    model.fit(X_train, (t_train, e_train), X_val=X_test, durations_val=t_test, events_val=e_test)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Evaluate on test set
    print("\nPredicting risks...")
    risk = model.predict_risk(X_test)
    c_index = model.score(X_test, (t_test, e_test))
    
    print(f"\n✓ Concordance Index: {c_index:.4f}")
    print(f"  (Expected: >0.65 for good model, 0.5 = random, your model: {c_index:.4f})")
    
    # Compare with baseline (MLP without TabICL)
    print("\n" + "-"*60)
    print("BASELINE COMPARISON (MLP without TabICL)")
    print("-"*60)
    
    baseline = TabICLSurvivor(
        backbone="mlp",  # Simple MLP on standardized features
        hidden=128,
        epochs=100,
        patience=15,
        verbose=False
    )
    
    print("Fitting baseline model...")
    baseline.fit(X_train, (t_train, e_train), X_val=X_test, durations_val=t_test, events_val=e_test)
    baseline_c_index = baseline.score(X_test, (t_test, e_test))
    
    print(f"\n✓ Baseline C-Index: {baseline_c_index:.4f}")
    print(f"✓ TabICL C-Index:   {c_index:.4f}")
    print(f"✓ Improvement:      {c_index - baseline_c_index:+.4f}")
    
    # Compare with XGBoost Cox survival model (max_depth=1)
    print("\n" + "-"*60)
    print("XGBOOST COX SURVIVAL (max_depth=1, decision stumps)")
    print("-"*60)
    
    # Prepare data for XGBoost Cox model (needs shape (n_samples, 1) for times/events)
    times_train = t_train.reshape(-1, 1)
    events_train = e_train.reshape(-1, 1)
    times_test = t_test.reshape(-1, 1)
    events_test = e_test.reshape(-1, 1)
    
    print("Fitting XGBoost Cox model...")
    xgb_models = train_xgb_cox_survival(
        X_train, 
        times_train, 
        events_train,
        val=(X_test, (times_test, events_test)),
        params={
            "seed": 42,
            "objective": 'survival:cox',
            "eval_metric": 'cox-nloglik',
            "tree_method": 'hist',
            "learning_rate": 0.05,
            "max_depth": 1,  # Decision stumps for fair comparison
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        num_boost_round=1000
    )
    
    # Predict risk (higher = higher hazard)
    xgb_risk = predict_xgb_cox(xgb_models, X_test)
    
    # Calculate C-index
    from src.tabicl.sklearn.survivor import _concordance_index
    xgb_c_index = _concordance_index(t_test, e_test, xgb_risk)
    
    print(f"\n✓ XGBoost C-Index:  {xgb_c_index:.4f}")
    print(f"✓ Baseline C-Index: {baseline_c_index:.4f}")
    print(f"✓ TabICL C-Index:   {c_index:.4f}")
    
    print(f"\nTabICL vs XGBoost:  {c_index - xgb_c_index:+.4f}")
    print(f"TabICL vs Baseline: {c_index - baseline_c_index:+.4f}")
    
    # Show risk correlation with true prognostic factors
    print("\n" + "-"*60)
    print("RISK CORRELATION WITH TRUE PROGNOSTIC FEATURES")
    print("-"*60)
    
    true_log_hazard = X_test @ true_beta
    corr_tabicl = np.corrcoef(risk, true_log_hazard)[0, 1]
    
    baseline_risk = baseline.predict_risk(X_test)
    corr_baseline = np.corrcoef(baseline_risk, true_log_hazard)[0, 1]
    
    corr_xgb = np.corrcoef(xgb_risk, true_log_hazard)[0, 1]
    
    print(f"Correlation with true log-hazard:")
    print(f"  TabICL:   {corr_tabicl:.4f}")
    print(f"  Baseline: {corr_baseline:.4f}")
    print(f"  XGBoost:  {corr_xgb:.4f}")
    
    # Predict survival functions for a few samples
    print("\n" + "-"*60)
    print("SURVIVAL FUNCTION PREDICTIONS")
    print("-"*60)
    
    S = model.predict_survival_function(X_test[:5], times=None, return_array=True)
    print(f"Generated survival curves for 5 test samples")
    print(f"Shape: {S.shape} (samples × time points)")
    print(f"Survival at median time: {S[:, S.shape[1]//2]}")
    
    # Final summary table
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\n{:<25} {:>10} {:>12} {:>10}".format("Model", "C-Index", "Correlation", "Rank"))
    print("-" * 60)
    
    results = [
        ("TabICL (frozen)", c_index, corr_tabicl),
        ("Baseline MLP", baseline_c_index, corr_baseline),
        ("XGBoost (depth=1)", xgb_c_index, corr_xgb),
    ]
    
    # Sort by C-index (descending)
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    
    for rank, (name, ci, corr) in enumerate(results_sorted, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print("{:<25} {:>10.4f} {:>12.4f} {:>10}".format(name, ci, corr, medal))
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("- C-Index > 0.65: Good discrimination")
    print("- C-Index > 0.75: Excellent discrimination") 
    print("- Correlation > 0.7: Strong relationship with true hazard")
    print("="*60)
