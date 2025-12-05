#!/usr/bin/env python
"""
Compute pairwise correlations for all models and human data
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Configuration (use relative paths from script directory)
OUTPUT_DIR = "./model_correlations"
MIN_SAMPLE_SIZE = 30

# Model files to process
MODEL_FILES = {
    'human': './merged_human_survey_philosophers.json',
    'llama3p18b': './merged_llama3p18b_philosophers.json',
    'mistral7b': './merged_mistral7b_philosophers.json',
    'gpt4o': './merged_openai_gpt4o_philosophers.json',
    'qwen3_4b': './merged_qwen3-4b_philosophers.json',
    'sonnet45': './merged_sonnet45_philosophers.json'
}

def load_data(filepath):
    """Load philosopher data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_response_matrix(data):
    """Create philosopher x question matrix"""

    # Collect all unique questions
    all_questions = set()
    for philosopher in data:
        if 'responses' in philosopher and philosopher['responses']:
            all_questions.update(philosopher['responses'].keys())

    all_questions = sorted(list(all_questions))

    # Create matrix
    matrix = []
    philosopher_names = []

    for philosopher in data:
        if 'responses' in philosopher and philosopher['responses']:
            row = []
            for question in all_questions:
                value = philosopher['responses'].get(question, np.nan)
                row.append(value)
            matrix.append(row)
            philosopher_names.append(philosopher['name'])

    df = pd.DataFrame(matrix, columns=all_questions, index=philosopher_names)

    return df

def compute_pairwise_correlations(df, min_sample_size=MIN_SAMPLE_SIZE):
    """Compute pairwise correlations between all questions"""

    questions = df.columns
    n_questions = len(questions)

    results = []

    for i in range(n_questions):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{n_questions}")

        for j in range(i + 1, n_questions):
            q1 = questions[i]
            q2 = questions[j]

            # Get values for both questions (remove NaN pairs)
            mask = df[q1].notna() & df[q2].notna()
            x = df.loc[mask, q1]
            y = df.loc[mask, q2]

            n = len(x)

            if n >= min_sample_size:
                # Compute Pearson correlation
                pearson_r, pearson_p = pearsonr(x, y)

                # Compute Spearman correlation
                spearman_r, spearman_p = spearmanr(x, y)

                results.append({
                    'question_1': q1,
                    'question_2': q2,
                    'n_samples': n,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'abs_pearson_r': abs(pearson_r),
                    'abs_spearman_r': abs(spearman_r)
                })

    return pd.DataFrame(results)

def summarize_correlations(corr_df, model_name):
    """Summarize correlation statistics"""

    summary = {
        'model': model_name,
        'n_correlations': len(corr_df),
        'n_with_data': len(corr_df[corr_df['pearson_r'].notna()]),
        'mean_sample_size': corr_df['n_samples'].mean(),
        'median_sample_size': corr_df['n_samples'].median(),
        'mean_pearson_r': corr_df['pearson_r'].mean(),
        'std_pearson_r': corr_df['pearson_r'].std(),
        'mean_abs_pearson_r': corr_df['abs_pearson_r'].mean(),
        'mean_spearman_r': corr_df['spearman_r'].mean(),
        'std_spearman_r': corr_df['spearman_r'].std(),
        'sig_001': (corr_df['pearson_p'] < 0.001).sum(),
        'sig_01': (corr_df['pearson_p'] < 0.01).sum(),
        'sig_05': (corr_df['pearson_p'] < 0.05).sum(),
        'strong_pos': (corr_df['pearson_r'] > 0.5).sum(),
        'strong_neg': (corr_df['pearson_r'] < -0.5).sum(),
    }

    return summary

def main():
    """Process all models"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("COMPUTING CORRELATIONS FOR ALL MODELS")
    print("="*70)

    summaries = []

    for model_name, filepath in MODEL_FILES.items():
        print(f"\n{model_name.upper()}")
        print("-"*70)

        # Check if file exists
        if not Path(filepath).exists():
            print(f"  ⚠ File not found: {filepath}")
            continue

        # Load data
        print(f"  Loading data...")
        data = load_data(filepath)
        print(f"  Loaded {len(data)} philosophers")

        # Create response matrix
        print(f"  Creating response matrix...")
        df = create_response_matrix(data)
        print(f"  Matrix shape: {df.shape[0]} philosophers × {df.shape[1]} questions")
        print(f"  Missing: {df.isna().sum().sum() / df.size * 100:.1f}%")

        # Compute correlations
        print(f"  Computing pairwise correlations (min n={MIN_SAMPLE_SIZE})...")
        corr_df = compute_pairwise_correlations(df, min_sample_size=MIN_SAMPLE_SIZE)
        print(f"  Computed {len(corr_df)} correlation pairs")

        # Save correlations
        output_file = f"{OUTPUT_DIR}/{model_name}_correlations.csv"
        corr_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")

        # Summarize
        summary = summarize_correlations(corr_df, model_name)
        summaries.append(summary)

        print(f"  Mean correlation: {summary['mean_pearson_r']:.3f}")
        print(f"  Strong positive (r>0.5): {summary['strong_pos']}")
        print(f"  Strong negative (r<-0.5): {summary['strong_neg']}")

    # Save summary comparison
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_file = f"{OUTPUT_DIR}/summary_comparison.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")

        # Print comparison table
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print()
        print(summary_df[['model', 'n_correlations', 'mean_pearson_r',
                          'mean_abs_pearson_r', 'strong_pos', 'strong_neg']].to_string(index=False))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
