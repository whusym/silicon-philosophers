#!/usr/bin/env python3
"""
Compare base vs fine-tuned model performance.

This script generates comparison tables between the base model and 
fine-tuned (DPO) model to evaluate whether fine-tuning improved 
alignment with human philosopher responses.

Requires:
- model_correlations/quality_metrics_corrected.csv (from 5a_compute_quality_metrics.py)

Outputs:
- Console output with comparison tables
- finetuning_comparison.txt (text report)
"""

import pandas as pd
import os

# Configuration
METRICS_FILE = "model_correlations/quality_metrics_corrected.csv"
OUTPUT_FILE = "finetuning_comparison.txt"

# Models to compare (base vs fine-tuned)
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # or "qwen2.5_0.5b" etc
FINETUNED_MODEL = "qwen2.5_0.5b_philosopher_dpo"  # match your naming convention


def load_metrics(filepath):
    """Load metrics CSV"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        print("Please run 5a_compute_quality_metrics.py first")
        return None
    return pd.read_csv(filepath)


def find_best_match(available_models, target, keywords):
    """Find best matching model from available options"""
    for model in available_models:
        if target.lower() in model.lower():
            return model
    for keyword in keywords:
        for model in available_models:
            if keyword.lower() in model.lower():
                return model
    return None


def format_value(val, is_percentage=False, lower_better=False):
    """Format a value with direction indicator"""
    if pd.isna(val):
        return "N/A"
    if is_percentage:
        return f"{val:.1f}%"
    return f"{val:.4f}"


def print_comparison_table(df, base_model, finetuned_model, human_model='human'):
    """Print detailed comparison table"""
    output_lines = []
    
    def log(line=""):
        print(line)
        output_lines.append(line)
    
    log("=" * 90)
    log("FINE-TUNING COMPARISON REPORT")
    log("=" * 90)
    log("")
    
    available_models = df['model'].tolist()
    log(f"Available models: {', '.join(available_models)}")
    log("")
    
    # Find models
    base = find_best_match(available_models, base_model, ['llama', 'qwen', 'mistral'])
    finetuned = find_best_match(available_models, finetuned_model, ['finetuned', 'ft', 'dpo'])
    human = 'human' if 'human' in available_models else None
    
    if not base:
        log(f"Warning: No base model found matching '{base_model}'")
        if available_models:
            base = available_models[0]
            log(f"Using: {base}")
    
    log("")
    log("-" * 90)
    log("COMPARISON CONFIGURATION")
    log("-" * 90)
    log(f"  Base Model:      {base or 'N/A'}")
    log(f"  Fine-tuned:      {finetuned or 'N/A'}")
    log(f"  Human Reference: {human or 'N/A'}")
    log("")
    
    # Get data rows
    base_row = df[df['model'] == base].iloc[0] if base else None
    ft_row = df[df['model'] == finetuned].iloc[0] if finetuned else None
    human_row = df[df['model'] == human].iloc[0] if human else None
    
    # Define metrics to compare
    metrics = [
        ('pct_usable', '% Usable Responses', True, False),  # (col, name, is_pct, lower_better)
        ('avg_entropy', 'Average Entropy', False, False),
        ('avg_effective_categories', 'Effective Categories', False, False),
        ('avg_concentration', 'Response Concentration', False, True),
        ('kl_divergence_mean', 'KL Divergence (vs Human)', False, True),
        ('js_divergence_mean', 'JS Divergence (vs Human)', False, True),
    ]
    
    log("-" * 90)
    log("METRIC COMPARISON TABLE")
    log("-" * 90)
    log("")
    
    # Table header
    header = f"{'Metric':<35} {'Base':>12} {'Finetuned':>12} {'Human':>12} {'Change':>12} {'Better?':>10}"
    log(header)
    log("-" * 90)
    
    improvements = 0
    regressions = 0
    
    for col, name, is_pct, lower_better in metrics:
        base_val = base_row.get(col) if base_row is not None and col in base_row else None
        ft_val = ft_row.get(col) if ft_row is not None and col in ft_row else None
        human_val = human_row.get(col) if human_row is not None and col in human_row else None
        
        base_str = format_value(base_val, is_pct)
        ft_str = format_value(ft_val, is_pct) if finetuned else "N/A"
        human_str = format_value(human_val, is_pct) if human else "N/A"
        
        # Calculate change
        if base_val is not None and ft_val is not None and not pd.isna(base_val) and not pd.isna(ft_val):
            change = ft_val - base_val
            if is_pct:
                change_str = f"{change:+.1f}%"
            else:
                change_str = f"{change:+.4f}"
            
            # Determine if improvement
            if lower_better:
                is_better = change < 0
            else:
                is_better = change > 0
            
            better_str = "✓ YES" if is_better else "✗ NO"
            
            if is_better:
                improvements += 1
            else:
                regressions += 1
        else:
            change_str = "N/A"
            better_str = "-"
        
        row = f"{name:<35} {base_str:>12} {ft_str:>12} {human_str:>12} {change_str:>12} {better_str:>10}"
        log(row)
    
    log("-" * 90)
    log("")
    
    # Summary
    log("-" * 90)
    log("SUMMARY")
    log("-" * 90)
    
    if finetuned:
        total_metrics = improvements + regressions
        log(f"  Metrics Improved:  {improvements}/{total_metrics}")
        log(f"  Metrics Regressed: {regressions}/{total_metrics}")
        
        if improvements > regressions:
            log("\n  ✓ Fine-tuning shows OVERALL IMPROVEMENT")
        elif regressions > improvements:
            log("\n  ✗ Fine-tuning shows OVERALL REGRESSION")
        else:
            log("\n  ~ Fine-tuning shows MIXED RESULTS")
    else:
        log("  No fine-tuned model found for comparison.")
        log("  Run 7_finetune_dpo.py to create a fine-tuned model,")
        log("  then re-run 5c_compute_all_model_correlations.py")
    
    log("")
    
    # Model ranking by KL divergence
    log("-" * 90)
    log("MODEL RANKING BY KL DIVERGENCE (lower = more similar to humans)")
    log("-" * 90)
    
    if 'kl_divergence_mean' in df.columns:
        ranked = df[df['model'] != 'human'].dropna(subset=['kl_divergence_mean'])
        ranked = ranked.sort_values('kl_divergence_mean')
        
        log("")
        log(f"{'Rank':<6} {'Model':<40} {'KL Divergence':>15}")
        log("-" * 65)
        
        for idx, row in enumerate(ranked.itertuples(), 1):
            marker = " *" if row.model == finetuned else ""
            log(f"{idx:<6} {row.model:<40} {row.kl_divergence_mean:>15.4f}{marker}")
        
        log("")
        log("  * = Fine-tuned model")
    else:
        log("  KL divergence data not available")
    
    log("")
    log("=" * 90)
    log("END OF REPORT")
    log("=" * 90)
    
    return "\n".join(output_lines)


def print_all_models_table(df):
    """Print comparison table for all available models"""
    output_lines = []
    
    def log(line=""):
        print(line)
        output_lines.append(line)
    
    log("")
    log("=" * 110)
    log("ALL MODELS COMPARISON")
    log("=" * 110)
    log("")
    
    # Select columns to display
    display_cols = ['model', 'pct_usable', 'avg_entropy', 'avg_effective_categories', 
                    'avg_concentration', 'kl_divergence_mean', 'js_divergence_mean']
    
    available_cols = [c for c in display_cols if c in df.columns]
    
    # Create formatted table
    headers = {
        'model': 'Model',
        'pct_usable': '% Usable',
        'avg_entropy': 'Entropy',
        'avg_effective_categories': 'Eff. Cat.',
        'avg_concentration': 'Conc.',
        'kl_divergence_mean': 'KL Div',
        'js_divergence_mean': 'JS Div'
    }
    
    header_line = f"{'Model':<35}"
    for col in available_cols[1:]:
        header_line += f" {headers.get(col, col):>12}"
    log(header_line)
    log("-" * 110)
    
    for _, row in df.iterrows():
        line = f"{row['model']:<35}"
        for col in available_cols[1:]:
            val = row.get(col)
            if pd.isna(val):
                line += f" {'N/A':>12}"
            elif col == 'pct_usable':
                line += f" {val:>11.1f}%"
            else:
                line += f" {val:>12.4f}"
        log(line)
    
    log("-" * 110)
    log("")
    
    return "\n".join(output_lines)


def main():
    print("=" * 90)
    print("FINE-TUNING COMPARISON ANALYSIS")
    print("=" * 90)
    
    # Load metrics
    df = load_metrics(METRICS_FILE)
    if df is None:
        return
    
    print(f"\nLoaded metrics for {len(df)} models")
    
    # Print comparison table
    report = print_comparison_table(df, BASE_MODEL, FINETUNED_MODEL)
    
    # Print all models table
    all_models_report = print_all_models_table(df)
    
    # Save report to file
    full_report = report + "\n" + all_models_report
    with open(OUTPUT_FILE, 'w') as f:
        f.write(full_report)
    
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
