#!/usr/bin/env python
"""
Compute comprehensive quality metrics for LLM responses
Captures: variation quality, information content, and correlation reliability

Metrics computed:
- Shannon entropy and effective categories
- Quality score (composite metric)
- RV coefficient (structural similarity)
- Mantel test (correlation matrix similarity)
- KL/JS divergence (distribution comparison)
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import rel_entr

# Configuration - use relative paths from script directory
# These can be overridden by setting environment variables or editing this dict
MODEL_FILES = {
    'human': './merged_human_survey_philosophers.json',
    'llama3p1_8b': './merged_llama3p18b_philosophers.json',
    'llama3p1_8b_finetuned': './merged_llama3p18b_philosophers_finetuned.json',
    'mistral7b': './merged_mistral7b_philosophers.json',
    'gpt4o': './merged_openai_gpt4o_philosophers.json',
    'gpt51': './merged_gpt51_philosophers.json',
    'qwen3_4b': './merged_qwen3-4b_philosophers.json',
    'sonnet45': './merged_sonnet45_philosophers.json',
}

OUTPUT_DIR = './model_correlations'


def load_to_dataframe(filepath):
    """Convert JSON to DataFrame"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract responses
    rows = []
    for phil in data:
        row = {'philosopher': phil.get('name', 'Unknown')}
        row.update(phil.get('responses', {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index('philosopher')

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def compute_entropy(series):
    """Compute Shannon entropy for a series"""
    series = series.dropna()
    if len(series) < 2:
        return 0.0

    # Get value counts and probabilities
    counts = series.value_counts()
    probs = counts / len(series)

    # Shannon entropy: -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))

    return entropy

def compute_effective_categories(series):
    """Compute effective number of categories (exponential of entropy)"""
    entropy = compute_entropy(series)
    return 2 ** entropy

def compute_question_quality_metrics(df, min_n=30):
    """Compute quality metrics for each question"""

    metrics = []

    for col in df.columns:
        values = df[col].dropna()
        n = len(values)

        if n < min_n:
            continue

        # Basic statistics
        variance = values.var()
        std = values.std()
        unique_vals = len(values.unique())

        # Entropy measures
        entropy = compute_entropy(values)
        effective_cats = compute_effective_categories(values)
        max_entropy = np.log2(unique_vals) if unique_vals > 1 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Distribution shape
        value_counts = values.value_counts()
        max_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
        concentration = max_freq / n  # How concentrated on single value

        # Coefficient of variation (relative variability)
        mean_val = values.mean()
        cv = std / mean_val if mean_val != 0 else 0

        metrics.append({
            'question': col,
            'n': n,
            'unique_values': unique_vals,
            'variance': variance,
            'std': std,
            'mean': mean_val,
            'cv': cv,  # Coefficient of variation
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'effective_categories': effective_cats,
            'concentration': concentration,  # % in most common value
            'is_zero_var': variance < 1e-10,
            'is_low_var': unique_vals <= 2,
        })

    return pd.DataFrame(metrics)

def compute_aggregate_metrics(question_metrics):
    """Compute aggregate quality metrics across all questions"""

    n_questions = len(question_metrics)

    if n_questions == 0:
        return {}

    # Proportion of problematic questions
    pct_zero_var = (question_metrics['is_zero_var'].sum() / n_questions) * 100
    pct_low_var = (question_metrics['is_low_var'].sum() / n_questions) * 100
    pct_problematic = ((question_metrics['is_zero_var'] | question_metrics['is_low_var']).sum() / n_questions) * 100

    # Average information content
    avg_entropy = question_metrics['entropy'].mean()
    avg_normalized_entropy = question_metrics['normalized_entropy'].mean()
    avg_effective_cats = question_metrics['effective_categories'].mean()

    # Average variation
    avg_variance = question_metrics['variance'].mean()
    avg_std = question_metrics['std'].mean()
    avg_cv = question_metrics['cv'].mean()
    avg_unique = question_metrics['unique_values'].mean()

    # Average concentration (lower is better)
    avg_concentration = question_metrics['concentration'].mean()

    # Quality score (composite metric)
    # Normalize components to 0-100 scale (higher is better)
    entropy_score = avg_normalized_entropy * 100  # Already 0-1
    variance_score = min(avg_variance / 0.25 * 100, 100)  # Scale: 0.25 variance = 100
    diversity_score = min((avg_unique - 1) / 4 * 100, 100)  # Scale: 5 unique = 100
    concentration_penalty = (1 - avg_concentration) * 100  # Invert: low concentration = high score

    # Weighted composite score
    quality_score = (
        0.30 * entropy_score +
        0.25 * variance_score +
        0.20 * diversity_score +
        0.25 * concentration_penalty
    )

    # Effective usable questions (after filtering problematic ones)
    n_usable = (~(question_metrics['is_zero_var'] | question_metrics['is_low_var'])).sum()
    pct_usable = (n_usable / n_questions) * 100

    return {
        'n_questions': n_questions,
        'n_usable': n_usable,
        'pct_usable': pct_usable,
        'pct_zero_var': pct_zero_var,
        'pct_low_var': pct_low_var,
        'pct_problematic': pct_problematic,
        'avg_entropy': avg_entropy,
        'avg_normalized_entropy': avg_normalized_entropy,
        'avg_effective_categories': avg_effective_cats,
        'avg_variance': avg_variance,
        'avg_std': avg_std,
        'avg_cv': avg_cv,
        'avg_unique_values': avg_unique,
        'avg_concentration': avg_concentration,
        'quality_score': quality_score,
    }

def compute_correlation_reliability_metrics(df, min_n=30):
    """Compute metrics for correlation reliability"""

    # Get questions with sufficient data
    valid_cols = [col for col in df.columns if df[col].notna().sum() >= min_n]

    if len(valid_cols) < 2:
        return {}

    df_valid = df[valid_cols]

    # Compute correlation matrix
    corr_matrix = df_valid.corr(method='pearson', min_periods=min_n)

    # Extract upper triangle (avoid duplicates)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = corr_matrix.where(mask).stack().values
    correlations = correlations[~np.isnan(correlations)]

    if len(correlations) == 0:
        return {}

    # Correlation statistics
    mean_abs_r = np.abs(correlations).mean()
    median_abs_r = np.median(np.abs(correlations))
    std_r = np.std(correlations)

    # Distribution characteristics
    pct_strong_pos = (correlations > 0.5).sum() / len(correlations) * 100
    pct_strong_neg = (correlations < -0.5).sum() / len(correlations) * 100
    pct_extreme = ((np.abs(correlations) > 0.5).sum() / len(correlations)) * 100

    # Signal-to-noise ratio
    # Signal: mean absolute correlation
    # Noise: standard deviation
    snr = mean_abs_r / std_r if std_r > 0 else 0

    return {
        'n_correlations': len(correlations),
        'mean_abs_r': mean_abs_r,
        'median_abs_r': median_abs_r,
        'std_r': std_r,
        'pct_strong_pos': pct_strong_pos,
        'pct_strong_neg': pct_strong_neg,
        'pct_extreme': pct_extreme,
        'signal_to_noise': snr,
    }

def compute_distribution_divergence(model_df, human_df):
    """Compute KL divergence between model and human response distributions"""

    # Get common questions
    common_questions = set(model_df.columns) & set(human_df.columns)

    if len(common_questions) == 0:
        return {'n_common': 0, 'avg_kl_divergence': np.nan}

    kl_divergences = []
    js_divergences = []

    for q in common_questions:
        model_vals = model_df[q].dropna()
        human_vals = human_df[q].dropna()

        if len(model_vals) < 10 or len(human_vals) < 10:
            continue

        # Get value counts
        all_values = sorted(set(model_vals) | set(human_vals))

        model_counts = pd.Series([sum(model_vals == v) for v in all_values], index=all_values)
        human_counts = pd.Series([sum(human_vals == v) for v in all_values], index=all_values)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        model_probs = (model_counts + epsilon) / (model_counts + epsilon).sum()
        human_probs = (human_counts + epsilon) / (human_counts + epsilon).sum()

        # KL divergence: KL(human || model)
        kl = np.sum(rel_entr(human_probs.values, model_probs.values))
        kl_divergences.append(kl)

        # Jensen-Shannon divergence (symmetric)
        m = (human_probs + model_probs) / 2
        js = 0.5 * np.sum(rel_entr(human_probs.values, m.values)) + \
             0.5 * np.sum(rel_entr(model_probs.values, m.values))
        js_divergences.append(js)

    return {
        'n_common': len(common_questions),
        'n_compared': len(kl_divergences),
        'avg_kl_divergence': np.mean(kl_divergences) if kl_divergences else np.nan,
        'avg_js_divergence': np.mean(js_divergences) if js_divergences else np.nan,
    }

def compute_question_correlation_matrix(df, min_common=30):
    """Compute correlation matrix between questions"""
    return df.corr(method='pearson', min_periods=min_common)

def rv_coefficient(A, B):
    """
    RV coefficient: measures similarity between two covariance matrices
    Range [0, 1], with 1 = identical structure
    """
    # Only compare common questions
    common_idx = A.index.intersection(B.index)
    
    if len(common_idx) < 2:
        return np.nan
        
    A_sub = A.loc[common_idx, common_idx].values
    B_sub = B.loc[common_idx, common_idx].values
    
    # Remove NaNs
    if np.any(np.isnan(A_sub)) or np.any(np.isnan(B_sub)):
        A_sub = np.nan_to_num(A_sub, nan=0)
        B_sub = np.nan_to_num(B_sub, nan=0)
        
    # RV = trace(A @ B) / sqrt(trace(A @ A) * trace(B @ B))
    numerator = np.trace(A_sub @ B_sub)
    denominator = np.sqrt(np.trace(A_sub @ A_sub) * np.trace(B_sub @ B_sub))
    
    if denominator == 0:
        return np.nan
        
    return numerator / denominator

def mantel_test(A, B, permutations=1000):
    """
    Mantel test: statistical test for correlation between two matrices
    Returns: (correlation, p_value)
    """
    # Only compare common questions
    common_idx = A.index.intersection(B.index)
    
    if len(common_idx) < 2:
        return np.nan, np.nan
        
    A_sub = A.loc[common_idx, common_idx]
    B_sub = B.loc[common_idx, common_idx]
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(A_sub, dtype=bool), k=1)
    
    a_vals = A_sub.values[mask]
    b_vals = B_sub.values[mask]
    
    # Remove NaN pairs
    valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
    a_vals = a_vals[valid]
    b_vals = b_vals[valid]
    
    if len(a_vals) < 10:
        return np.nan, np.nan
        
    # Observed correlation
    obs_corr, _ = stats.pearsonr(a_vals, b_vals)
    
    # Permutation test
    n = len(a_vals)
    perm_corrs = []
    
    rng = np.random.RandomState(42)
    
    for _ in range(permutations):
        perm_idx = rng.permutation(n)
        perm_corr, _ = stats.pearsonr(a_vals[perm_idx], b_vals)
        perm_corrs.append(perm_corr)
        
    # P-value (two-tailed)
    p_value = np.sum(np.abs(perm_corrs) >= np.abs(obs_corr)) / permutations
    
    return obs_corr, p_value

def compute_structural_similarity(model_df, human_df, min_n=30):
    """Compute structural similarity metrics (RV, Mantel)"""
    
    # Compute correlation matrices
    model_corr = compute_question_correlation_matrix(model_df, min_common=min_n)
    human_corr = compute_question_correlation_matrix(human_df, min_common=min_n)
    
    if model_corr.empty or human_corr.empty:
        return {
            'rv_coefficient': np.nan,
            'mantel_corr': np.nan,
            'mantel_p': np.nan
        }
    
    # RV Coefficient
    rv = rv_coefficient(model_corr, human_corr)
    
    # Mantel Test
    mantel_r, mantel_p = mantel_test(model_corr, human_corr)
    
    return {
        'rv_coefficient': rv,
        'mantel_corr': mantel_r,
        'mantel_p': mantel_p
    }

def main():
    print("="*80)
    print("COMPREHENSIVE QUALITY METRICS")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all data (skip missing files)
    print("\nLoading data...")
    dataframes = {}
    for model, filepath in MODEL_FILES.items():
        if os.path.exists(filepath):
            dataframes[model] = load_to_dataframe(filepath)
            print(f"  ✓ {model}: {len(dataframes[model])} philosophers, {len(dataframes[model].columns)} questions")
        else:
            print(f"  ⚠ {model}: File not found ({filepath})")

    if not dataframes:
        print("\nNo data files found. Please ensure the merged philosopher files exist.")
        return

    # Compute metrics for each model
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    all_metrics = []
    question_metrics_all = {}

    human_df = dataframes.get('human')

    for model, df in dataframes.items():
        print(f"\n{model.upper()}:")

        # Question-level metrics
        q_metrics = compute_question_quality_metrics(df)
        question_metrics_all[model] = q_metrics

        # Aggregate metrics
        agg_metrics = compute_aggregate_metrics(q_metrics)
        agg_metrics['model'] = model

        # Correlation reliability
        corr_metrics = compute_correlation_reliability_metrics(df)
        agg_metrics.update(corr_metrics)

        # Distribution divergence (compare to human)
        if model != 'human' and human_df is not None:
            div_metrics = compute_distribution_divergence(df, human_df)
            agg_metrics.update(div_metrics)
            
            # Structural similarity (compare to human)
            struct_metrics = compute_structural_similarity(df, human_df)
            agg_metrics.update(struct_metrics)

        all_metrics.append(agg_metrics)

        # Print key metrics
        print(f"  Quality Score: {agg_metrics.get('quality_score', 0):.2f}/100")
        print(f"  Usable Questions: {agg_metrics.get('pct_usable', 0):.1f}%")
        print(f"  Avg Entropy: {agg_metrics.get('avg_entropy', 0):.3f}")
        print(f"  Avg Effective Categories: {agg_metrics.get('avg_effective_categories', 0):.2f}")
        print(f"  Signal-to-Noise Ratio: {agg_metrics.get('signal_to_noise', 0):.3f}")
        
        if model != 'human':
            print(f"  RV Coefficient: {agg_metrics.get('rv_coefficient', 0):.3f}")
            print(f"  Mantel r: {agg_metrics.get('mantel_corr', 0):.3f} (p={agg_metrics.get('mantel_p', 1.0):.3f})")

    # Create comparison DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "quality_metrics.csv")
    metrics_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved metrics to: {output_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("QUALITY METRICS COMPARISON")
    print("="*80)

    # Select key columns
    comparison_cols = [
        'model', 'quality_score', 'pct_usable', 'avg_entropy',
        'avg_effective_categories', 'signal_to_noise', 
        'rv_coefficient', 'mantel_corr', 'avg_kl_divergence'
    ]
    
    # Filter columns that exist
    comparison_cols = [c for c in comparison_cols if c in metrics_df.columns]

    comparison_df = metrics_df[comparison_cols].copy()
    comparison_df = comparison_df.sort_values('quality_score', ascending=False)

    print("\n" + comparison_df.to_string(index=False))

    # Rankings
    print("\n" + "="*80)
    print("MODEL RANKINGS")
    print("="*80)

    print("\nBy Quality Score (higher is better):")
    for i, row in comparison_df.iterrows():
        rank = comparison_df.index.get_loc(i) + 1
        print(f"  {rank}. {row['model']:15s} {row['quality_score']:6.2f}/100")

    print("\nBy Usable Questions % (higher is better):")
    ranked = metrics_df.sort_values('pct_usable', ascending=False)
    for i, (idx, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['model']:15s} {row['pct_usable']:6.2f}%")

    print("\nBy Average Entropy (higher is better - more information):")
    ranked = metrics_df.sort_values('avg_entropy', ascending=False)
    for i, (idx, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['model']:15s} {row['avg_entropy']:6.3f} bits")

if __name__ == "__main__":
    main()
