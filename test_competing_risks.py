"""
Test script for TabICLCompetingRisks implementation.

Generates synthetic competing risks data and evaluates the model.
"""

import numpy as np
from src.tabicl.sklearn.competing_risks import TabICLCompetingRisks


def generate_competing_risks_data(n, n_features=10, n_event_types=2, random_state=42):
    """
    Generate synthetic competing risks data with known prognostic factors.
    
    Parameters
    ----------
    n : int
        Number of samples
    n_features : int
        Number of features
    n_event_types : int
        Number of competing event types
    random_state : int
        Random seed
    
    Returns
    -------
    X : np.ndarray
        Features
    times : np.ndarray
        Observed times
    event_types : np.ndarray
        Event types (0=censored, 1,2,...,K=events)
    true_betas : list of np.ndarray
        True coefficients for each event type
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n, n_features)
    
    # Define different prognostic factors for each event type
    true_betas = []
    event_times = []
    
    for k in range(n_event_types):
        # Each event type has different prognostic features
        beta_k = np.zeros(n_features)
        # Assign different features to different causes
        start_idx = k * (n_features // n_event_types)
        end_idx = min((k + 1) * (n_features // n_event_types), n_features)
        beta_k[start_idx:end_idx] = np.random.uniform(-0.8, 0.8, end_idx - start_idx)
        
        true_betas.append(beta_k)
        
        # Generate times for this event type
        log_hazard_k = X @ beta_k
        hazard_ratios_k = np.exp(log_hazard_k)
        times_k = np.random.exponential(100 / hazard_ratios_k)
        event_times.append(times_k)
    
    # Generate censoring times
    censoring_times = np.random.exponential(150, size=n)
    
    # Observed time is minimum across all event types and censoring
    all_times = np.stack(event_times + [censoring_times], axis=1)
    observed_times = np.min(all_times, axis=1)
    
    # Event type is which happened first
    event_types = np.argmin(all_times, axis=1)
    # 0, 1, ..., n_event_types-1, n_event_types (censoring)
    # Convert: 0->1, 1->2, ..., n_event_types->0 (censoring)
    event_types_final = np.where(event_types == n_event_types, 0, event_types + 1)
    
    return X, observed_times, event_types_final, true_betas


if __name__ == "__main__":
    print("="*70)
    print("TABICL COMPETING RISKS TEST")
    print("="*70)
    
    # Generate data
    n_train, n_test = 600, 150
    n_features = 10
    n_event_types = 2
    
    print(f"\nGenerating competing risks data...")
    print(f"- Training samples: {n_train}")
    print(f"- Test samples: {n_test}")
    print(f"- Features: {n_features}")
    print(f"- Event types: {n_event_types}")
    
    X_train, t_train, e_train, true_betas = generate_competing_risks_data(
        n_train, n_features, n_event_types, random_state=42
    )
    X_test, t_test, e_test, _ = generate_competing_risks_data(
        n_test, n_features, n_event_types, random_state=43
    )
    
    # Print data statistics
    print(f"\nTraining data statistics:")
    for k in range(1, n_event_types + 1):
        n_events_k = (e_train == k).sum()
        print(f"  Event type {k}: {n_events_k} events ({100*n_events_k/n_train:.1f}%)")
    n_censored = (e_train == 0).sum()
    print(f"  Censored: {n_censored} ({100*n_censored/n_train:.1f}%)")
    
    print(f"\nTrue prognostic coefficients:")
    for k, beta_k in enumerate(true_betas, 1):
        non_zero = np.where(np.abs(beta_k) > 0.01)[0]
        print(f"  Cause {k}: Features {non_zero.tolist()} with coefs {beta_k[non_zero]}")
    
    # Train TabICL model
    print("\n" + "="*70)
    print("TRAINING TABICL COMPETING RISKS MODEL")
    print("="*70)
    
    model_tabicl = TabICLCompetingRisks(
        n_event_types=n_event_types,
        backbone="tabicl",
        checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
        hidden=128,
        epochs=100,
        patience=15,
        verbose=True
    )
    
    print("\nFitting TabICL model...")
    model_tabicl.fit(
        X_train, (t_train, e_train),
        X_val=X_test, y_val=(t_test, e_test)
    )
    
    # Train baseline MLP model
    print("\n" + "="*70)
    print("TRAINING BASELINE MLP MODEL")
    print("="*70)
    
    model_baseline = TabICLCompetingRisks(
        n_event_types=n_event_types,
        backbone="mlp",
        hidden=128,
        epochs=100,
        patience=15,
        verbose=False
    )
    
    print("\nFitting baseline model...")
    model_baseline.fit(
        X_train, (t_train, e_train),
        X_val=X_test, y_val=(t_test, e_test)
    )
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Predict cause-specific risks
    risks_tabicl = model_tabicl.predict_cause_specific_risk(X_test)
    risks_baseline = model_baseline.predict_cause_specific_risk(X_test)
    
    print(f"\nPredicted risks shape: {risks_tabicl.shape}")
    
    # Compute C-indices
    print("\nCause-specific C-indices:")
    print(f"{'Cause':<10} {'TabICL':<12} {'Baseline':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for k in range(1, n_event_types + 1):
        c_tabicl = model_tabicl.score(X_test, (t_test, e_test), cause=k)
        c_baseline = model_baseline.score(X_test, (t_test, e_test), cause=k)
        
        if not np.isnan(c_tabicl):
            print(f"Cause {k:<5} {c_tabicl:<12.4f} {c_baseline:<12.4f} {c_tabicl - c_baseline:+.4f}")
        else:
            print(f"Cause {k:<5} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    c_avg_tabicl = model_tabicl.score(X_test, (t_test, e_test))
    c_avg_baseline = model_baseline.score(X_test, (t_test, e_test))
    
    print("-" * 50)
    print(f"{'Average':<10} {c_avg_tabicl:<12.4f} {c_avg_baseline:<12.4f} {c_avg_tabicl - c_avg_baseline:+.4f}")
    
    # Correlation with true hazards
    print("\nCorrelation with true log-hazards:")
    print(f"{'Cause':<10} {'TabICL':<12} {'Baseline':<12}")
    print("-" * 40)
    
    for k in range(n_event_types):
        true_log_hazard_k = X_test @ true_betas[k]
        corr_tabicl = np.corrcoef(risks_tabicl[:, k], true_log_hazard_k)[0, 1]
        corr_baseline = np.corrcoef(risks_baseline[:, k], true_log_hazard_k)[0, 1]
        
        print(f"Cause {k+1:<5} {corr_tabicl:<12.4f} {corr_baseline:<12.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTabICL Competing Risks achieves:")
    print(f"  - Average C-index: {c_avg_tabicl:.4f}")
    print(f"  - Improvement over MLP: {c_avg_tabicl - c_avg_baseline:+.4f}")
    
    if c_avg_tabicl > 0.65:
        print(f"  - ✓ Good discrimination (C-index > 0.65)")
    if c_avg_tabicl > 0.75:
        print(f"  - ✓ Excellent discrimination (C-index > 0.75)")
    
    print("\n" + "="*70)

