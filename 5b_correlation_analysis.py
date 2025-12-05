#!/usr/bin/env python3
"""
Correlation Analysis: Feature-Response Relationships

Analyzes statistical correlations between philosopher features (AOS, AOC, etc.)
and their survey responses. This approach reveals meaningful patterns even with sparse data.

Metrics used:
- Point-biserial correlation: Correlation between binary feature and response value
- Cohen's d effect size: Standardized difference in mean responses
- FDR-corrected p-values: Controls false discovery rate across many tests
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load the merged survey-philosophers data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def bin_year(year, bin_size=5):
    """Convert year to 5-year bin"""
    if year is None or pd.isna(year):
        return "Unknown"
    bin_start = (year // bin_size) * bin_size
    bin_end = bin_start + bin_size - 1
    return f"{bin_start}-{bin_end}"


def compute_feature_response_correlations(data, feature_type='areas_of_specialization', 
                                          min_feature_count=5, min_question_count=10):
    """
    Compute correlations between a feature type (AOS, AOC, etc.) and responses.
    
    Returns a DataFrame with correlation statistics for each (feature, question) pair.
    """
    results = []
    
    # Collect all unique features and questions
    all_features = set()
    all_questions = set()
    
    for phil in data:
        if 'responses' not in phil or not phil['responses']:
            continue
        features = phil.get(feature_type, []) or []
        all_features.update(features)
        all_questions.update(phil['responses'].keys())
    
    print(f"Found {len(all_features)} unique {feature_type}")
    print(f"Found {len(all_questions)} unique questions")
    
    # For each feature, for each question, compute correlation
    total_pairs = len(all_features) * len(all_questions)
    processed = 0
    
    for feature in all_features:
        for question in all_questions:
            processed += 1
            if processed % 10000 == 0:
                print(f"  Processing {processed}/{total_pairs} pairs...")
            
            # Collect responses from philosophers with and without this feature
            responses_with_feature = []
            responses_without_feature = []
            
            for phil in data:
                if 'responses' not in phil or not phil['responses']:
                    continue
                if question not in phil['responses']:
                    continue
                    
                response = phil['responses'][question]
                features = set(phil.get(feature_type, []) or [])
                
                if feature in features:
                    responses_with_feature.append(response)
                else:
                    responses_without_feature.append(response)
            
            # Skip if not enough data
            if len(responses_with_feature) < min_feature_count or len(responses_without_feature) < min_feature_count:
                continue
            
            n_with = len(responses_with_feature)
            n_without = len(responses_without_feature)
            mean_with = np.mean(responses_with_feature)
            mean_without = np.mean(responses_without_feature)
            
            # Point-biserial correlation
            all_responses = responses_with_feature + responses_without_feature
            feature_indicator = [1] * n_with + [0] * n_without
            
            try:
                corr, p_value = pointbiserialr(feature_indicator, all_responses)
            except:
                corr, p_value = np.nan, np.nan
            
            # Cohen's d effect size
            var_with = np.var(responses_with_feature, ddof=1) if n_with > 1 else 0
            var_without = np.var(responses_without_feature, ddof=1) if n_without > 1 else 0
            pooled_std = np.sqrt(
                ((n_with - 1) * var_with + (n_without - 1) * var_without) / 
                (n_with + n_without - 2)
            )
            if pooled_std > 0:
                cohens_d = (mean_with - mean_without) / pooled_std
            else:
                cohens_d = 0
            
            # Mann-Whitney U test (non-parametric alternative)
            try:
                _, mw_p_value = mannwhitneyu(responses_with_feature, responses_without_feature, 
                                             alternative='two-sided')
            except:
                mw_p_value = np.nan
            
            results.append({
                'feature': feature,
                'question': question,
                'feature_type': feature_type,
                'n_with_feature': n_with,
                'n_without_feature': n_without,
                'mean_with_feature': mean_with,
                'mean_without_feature': mean_without,
                'mean_diff': mean_with - mean_without,
                'point_biserial_corr': corr,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'mannwhitney_p': mw_p_value
            })
    
    df = pd.DataFrame(results)
    
    # Apply FDR correction for multiple testing
    if len(df) > 0:
        df['p_value_fdr'] = multipletests(df['p_value'].fillna(1), method='fdr_bh')[1]
        df['mw_p_value_fdr'] = multipletests(df['mannwhitney_p'].fillna(1), method='fdr_bh')[1]
    
    return df


def compute_categorical_correlations(data, feature_key, min_count=5):
    """Compute correlations for a categorical feature (country, year bin, etc.)."""
    results = []
    
    all_values = set()
    all_questions = set()
    
    for phil in data:
        if 'responses' not in phil or not phil['responses']:
            continue
        value = phil.get(feature_key)
        if value:
            if feature_key == 'year_of_phd_degree':
                value = bin_year(value)
            all_values.add(value)
        all_questions.update(phil['responses'].keys())
    
    print(f"Found {len(all_values)} unique {feature_key} values")
    
    for value in all_values:
        for question in all_questions:
            responses_with = []
            responses_without = []
            
            for phil in data:
                if 'responses' not in phil or not phil['responses']:
                    continue
                if question not in phil['responses']:
                    continue
                
                phil_value = phil.get(feature_key)
                if feature_key == 'year_of_phd_degree' and phil_value:
                    phil_value = bin_year(phil_value)
                
                response = phil['responses'][question]
                if phil_value == value:
                    responses_with.append(response)
                else:
                    responses_without.append(response)
            
            if len(responses_with) < min_count or len(responses_without) < min_count:
                continue
            
            n_with = len(responses_with)
            n_without = len(responses_without)
            mean_with = np.mean(responses_with)
            mean_without = np.mean(responses_without)
            
            all_responses = responses_with + responses_without
            indicator = [1] * n_with + [0] * n_without
            
            try:
                corr, p_value = pointbiserialr(indicator, all_responses)
            except:
                corr, p_value = np.nan, np.nan
            
            var_with = np.var(responses_with, ddof=1) if n_with > 1 else 0
            var_without = np.var(responses_without, ddof=1) if n_without > 1 else 0
            pooled_std = np.sqrt(
                ((n_with - 1) * var_with + (n_without - 1) * var_without) / 
                (n_with + n_without - 2)
            )
            cohens_d = (mean_with - mean_without) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'feature': value,
                'question': question,
                'feature_type': feature_key,
                'n_with_feature': n_with,
                'n_without_feature': n_without,
                'mean_with_feature': mean_with,
                'mean_without_feature': mean_without,
                'mean_diff': mean_with - mean_without,
                'point_biserial_corr': corr,
                'p_value': p_value,
                'cohens_d': cohens_d
            })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df['p_value_fdr'] = multipletests(df['p_value'].fillna(1), method='fdr_bh')[1]
    return df


def aggregate_feature_effects(correlations_df, p_threshold=0.05):
    """Aggregate effects by feature to see which features have consistent effects."""
    feature_stats = correlations_df.groupby('feature').agg({
        'point_biserial_corr': ['mean', 'std', 'count'],
        'cohens_d': ['mean', 'std'],
        'p_value_fdr': lambda x: (x < p_threshold).sum(),
        'mean_diff': 'mean',
        'n_with_feature': 'mean'
    }).reset_index()
    
    feature_stats.columns = ['feature', 'mean_corr', 'std_corr', 'n_questions', 
                             'mean_cohens_d', 'std_cohens_d', 'n_significant', 
                             'mean_response_diff', 'avg_n_with_feature']
    
    feature_stats['effect_consistency'] = feature_stats['mean_corr'].abs() / (feature_stats['std_corr'] + 0.01)
    feature_stats['significance_rate'] = feature_stats['n_significant'] / feature_stats['n_questions']
    
    return feature_stats


def aggregate_question_effects(correlations_df, p_threshold=0.05):
    """Aggregate effects by question to see which questions are most influenced by features."""
    question_stats = correlations_df.groupby('question').agg({
        'point_biserial_corr': ['mean', 'std', 'count', lambda x: x.abs().max()],
        'cohens_d': ['mean', lambda x: x.abs().max()],
        'p_value_fdr': lambda x: (x < p_threshold).sum(),
    }).reset_index()
    
    question_stats.columns = ['question', 'mean_corr', 'std_corr', 'n_features', 'max_abs_corr',
                              'mean_cohens_d', 'max_abs_cohens_d', 'n_significant']
    
    question_stats['significance_rate'] = question_stats['n_significant'] / question_stats['n_features']
    
    return question_stats


def main():
    print("="*80)
    print("CORRELATION ANALYSIS: Feature-Response Relationships")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data = load_data('merged_survey_philosophers.json')
    print(f"Loaded {len(data)} philosophers")
    
    philosophers_with_responses = [p for p in data if 'responses' in p and p['responses']]
    print(f"Philosophers with responses: {len(philosophers_with_responses)}")
    
    # 1. Areas of Specialization (AOS)
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Areas of Specialization (AOS)")
    print("="*80)
    aos_correlations = compute_feature_response_correlations(
        philosophers_with_responses, 
        feature_type='areas_of_specialization',
        min_feature_count=5,
        min_question_count=10
    )
    print(f"\nComputed {len(aos_correlations)} (feature, question) correlations")
    
    # Display strongest AOS correlations
    significant_aos = aos_correlations[aos_correlations['p_value_fdr'] < 0.05].copy()
    print(f"Significant correlations (FDR-corrected p < 0.05): {len(significant_aos)}")
    
    if len(significant_aos) > 0:
        significant_aos['abs_corr'] = significant_aos['point_biserial_corr'].abs()
        top_aos = significant_aos.nlargest(20, 'abs_corr')
        
        print("\nTop 20 Strongest Significant AOS Correlations:")
        print("-" * 120)
        print(f"{'Feature (AOS)':<45} {'Question':<45} {'Corr':>8} {'Cohen d':>8} {'p(FDR)':>10}")
        print("-" * 120)
        
        for _, row in top_aos.iterrows():
            feature_short = row['feature'][:43] if len(row['feature']) > 43 else row['feature']
            question_short = row['question'][:43] if len(row['question']) > 43 else row['question']
            print(f"{feature_short:<45} {question_short:<45} {row['point_biserial_corr']:>8.3f} {row['cohens_d']:>8.2f} {row['p_value_fdr']:>10.4f}")
    
    # 2. Areas of Interest (AOC)
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: Areas of Interest (AOC)")
    print("="*80)
    aoc_correlations = compute_feature_response_correlations(
        philosophers_with_responses, 
        feature_type='areas_of_interest',
        min_feature_count=5,
        min_question_count=10
    )
    print(f"\nComputed {len(aoc_correlations)} (feature, question) correlations")
    
    significant_aoc = aoc_correlations[aoc_correlations['p_value_fdr'] < 0.05].copy()
    print(f"Significant correlations (FDR-corrected p < 0.05): {len(significant_aoc)}")
    
    if len(significant_aoc) > 0:
        significant_aoc['abs_corr'] = significant_aoc['point_biserial_corr'].abs()
        top_aoc = significant_aoc.nlargest(20, 'abs_corr')
        
        print("\nTop 20 Strongest Significant AOC Correlations:")
        print("-" * 120)
        print(f"{'Feature (AOC)':<45} {'Question':<45} {'Corr':>8} {'Cohen d':>8} {'p(FDR)':>10}")
        print("-" * 120)
        
        for _, row in top_aoc.iterrows():
            feature_short = row['feature'][:43] if len(row['feature']) > 43 else row['feature']
            question_short = row['question'][:43] if len(row['question']) > 43 else row['question']
            print(f"{feature_short:<45} {question_short:<45} {row['point_biserial_corr']:>8.3f} {row['cohens_d']:>8.2f} {row['p_value_fdr']:>10.4f}")
    
    # 3. Aggregate feature effects
    print("\n" + "="*80)
    print("AGGREGATED FEATURE EFFECTS")
    print("="*80)
    
    aos_feature_stats = aggregate_feature_effects(aos_correlations)
    aos_feature_stats_sorted = aos_feature_stats.sort_values('n_significant', ascending=False)
    
    print("\nTop 15 AOS with Most Significant Correlations:")
    print("-" * 100)
    print(f"{'Feature':<50} {'Mean Corr':>10} {'N Quest':>8} {'N Signif':>8} {'Sig Rate':>8} {'Mean d':>8}")
    print("-" * 100)
    
    for _, row in aos_feature_stats_sorted.head(15).iterrows():
        feature_short = row['feature'][:48] if len(row['feature']) > 48 else row['feature']
        print(f"{feature_short:<50} {row['mean_corr']:>10.3f} {row['n_questions']:>8.0f} {row['n_significant']:>8.0f} {row['significance_rate']:>8.2%} {row['mean_cohens_d']:>8.2f}")
    
    aoc_feature_stats = aggregate_feature_effects(aoc_correlations)
    aoc_feature_stats_sorted = aoc_feature_stats.sort_values('n_significant', ascending=False)
    
    print("\nTop 15 AOC with Most Significant Correlations:")
    print("-" * 100)
    print(f"{'Feature':<50} {'Mean Corr':>10} {'N Quest':>8} {'N Signif':>8} {'Sig Rate':>8} {'Mean d':>8}")
    print("-" * 100)
    
    for _, row in aoc_feature_stats_sorted.head(15).iterrows():
        feature_short = row['feature'][:48] if len(row['feature']) > 48 else row['feature']
        print(f"{feature_short:<50} {row['mean_corr']:>10.3f} {row['n_questions']:>8.0f} {row['n_significant']:>8.0f} {row['significance_rate']:>8.2%} {row['mean_cohens_d']:>8.2f}")
    
    # 4. Question-level analysis
    print("\n" + "="*80)
    print("QUESTIONS MOST INFLUENCED BY PHILOSOPHER BACKGROUND")
    print("="*80)
    
    all_correlations = pd.concat([aos_correlations, aoc_correlations], ignore_index=True)
    question_stats = aggregate_question_effects(all_correlations)
    question_stats_sorted = question_stats.sort_values('n_significant', ascending=False)
    
    print("\nTop 20 Questions with Most Significant Feature Correlations:")
    print("-" * 110)
    print(f"{'Question':<60} {'N Signif':>8} {'Max |r|':>8} {'Max |d|':>8} {'Sig Rate':>8}")
    print("-" * 110)
    
    for _, row in question_stats_sorted.head(20).iterrows():
        q_short = row['question'][:58] if len(row['question']) > 58 else row['question']
        print(f"{q_short:<60} {row['n_significant']:>8.0f} {row['max_abs_corr']:>8.3f} {row['max_abs_cohens_d']:>8.2f} {row['significance_rate']:>8.2%}")
    
    # 5. PhD Country analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: PhD Country")
    print("="*80)
    country_correlations = compute_categorical_correlations(philosophers_with_responses, 'phd_country', min_count=5)
    print(f"Computed {len(country_correlations)} correlations")
    
    if len(country_correlations) > 0:
        country_sig = country_correlations[country_correlations['p_value_fdr'] < 0.05]
        print(f"Significant correlations: {len(country_sig)}")
        
        country_stats = aggregate_feature_effects(country_correlations)
        print("\nCorrelations by country:")
        for _, row in country_stats.sort_values('n_significant', ascending=False).head(10).iterrows():
            print(f"  {row['feature']:<20} n_sig={row['n_significant']:.0f}, mean_r={row['mean_corr']:.3f}")
    
    # 6. PhD Year analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: PhD Year (5-year bins)")
    print("="*80)
    year_correlations = compute_categorical_correlations(philosophers_with_responses, 'year_of_phd_degree', min_count=5)
    print(f"Computed {len(year_correlations)} correlations")
    
    if len(year_correlations) > 0:
        year_sig = year_correlations[year_correlations['p_value_fdr'] < 0.05]
        print(f"Significant correlations: {len(year_sig)}")
        
        year_stats = aggregate_feature_effects(year_correlations)
        print("\nCorrelations by year bin:")
        for _, row in year_stats.sort_values('feature').iterrows():
            if row['n_significant'] > 0:
                print(f"  {row['feature']:<15} n_sig={row['n_significant']:.0f}, mean_r={row['mean_corr']:+.3f}, mean_d={row['mean_cohens_d']:+.2f}")
    
    # 7. Summary statistics
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n1. OVERALL STATISTICS")
    print("-" * 60)
    print(f"   Total (feature, question) pairs analyzed: {len(all_correlations):,}")
    print(f"   Significant correlations (FDR < 0.05): {(all_correlations['p_value_fdr'] < 0.05).sum():,}")
    print(f"   Percentage significant: {(all_correlations['p_value_fdr'] < 0.05).mean():.1%}")
    
    print("\n2. EFFECT SIZE DISTRIBUTION")
    print("-" * 60)
    print(f"   Mean |correlation|: {all_correlations['point_biserial_corr'].abs().mean():.4f}")
    print(f"   Median |correlation|: {all_correlations['point_biserial_corr'].abs().median():.4f}")
    print(f"   Max |correlation|: {all_correlations['point_biserial_corr'].abs().max():.4f}")
    print(f"   Mean |Cohen's d|: {all_correlations['cohens_d'].abs().mean():.4f}")
    print(f"   Correlations with |r| > 0.2: {(all_correlations['point_biserial_corr'].abs() > 0.2).sum()}")
    print(f"   Correlations with |d| > 0.5 (medium effect): {(all_correlations['cohens_d'].abs() > 0.5).sum()}")
    
    print("\n3. BY FEATURE TYPE")
    print("-" * 60)
    aos_sig = (aos_correlations['p_value_fdr'] < 0.05).sum()
    aoc_sig = (aoc_correlations['p_value_fdr'] < 0.05).sum()
    print(f"   AOS: {len(aos_correlations):,} pairs, {aos_sig} significant ({aos_sig/len(aos_correlations):.1%})")
    print(f"   AOC: {len(aoc_correlations):,} pairs, {aoc_sig} significant ({aoc_sig/len(aoc_correlations):.1%})")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    all_correlations.to_csv('correlations_all.csv', index=False)
    aos_correlations.to_csv('correlations_aos.csv', index=False)
    aoc_correlations.to_csv('correlations_aoc.csv', index=False)
    country_correlations.to_csv('correlations_country.csv', index=False)
    year_correlations.to_csv('correlations_year.csv', index=False)
    aos_feature_stats.to_csv('feature_stats_aos.csv', index=False)
    aoc_feature_stats.to_csv('feature_stats_aoc.csv', index=False)
    question_stats.to_csv('question_stats.csv', index=False)
    
    print("Saved correlation results to CSV files:")
    print("  - correlations_all.csv")
    print("  - correlations_aos.csv")
    print("  - correlations_aoc.csv")
    print("  - correlations_country.csv")
    print("  - correlations_year.csv")
    print("  - feature_stats_aos.csv")
    print("  - feature_stats_aoc.csv")
    print("  - question_stats.csv")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()

