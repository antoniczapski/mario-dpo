"""
PCG Arena EDA - RQ3: What Do Tags Actually Mean?
=================================================
Hypotheses tested:
- H6: Tags correspond to measurable telemetry differences
- H7: Tag prediction from telemetry features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
EDA_DIR = Path(__file__).parent.parent
DATA_DIR = EDA_DIR / "00_data_preparation"
PLOTS_DIR = EDA_DIR / "plots"
RAW_DATA_DIR = EDA_DIR

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data():
    """Load prepared data."""
    level_stats = pd.read_csv(DATA_DIR / "level_stats_clean.csv")
    telemetry = pd.read_csv(DATA_DIR / "telemetry_flat.csv")
    
    # Load raw votes for tag analysis
    with open(RAW_DATA_DIR / "pcg-arena-votes-2026-01-28.json", 'r') as f:
        votes_raw = json.load(f)
    votes_df = pd.DataFrame(votes_raw)
    
    return level_stats, telemetry, votes_df

def h6_tag_telemetry_correspondence(level_stats, telemetry):
    """
    H6: Tags correspond to measurable telemetry differences.
    
    Test: Compare telemetry distributions across tag presence/absence.
    """
    print("\n" + "="*60)
    print("H6: Tag-Telemetry Correspondence")
    print("="*60)
    
    # Filter levels with sufficient data
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    # Define tags and corresponding expected telemetry patterns
    tag_expectations = {
        'tag_too_hard': {'metric': 'avg_deaths', 'expected': 'higher'},
        'tag_too_easy': {'metric': 'avg_deaths', 'expected': 'lower'},
        'tag_boring': {'metric': 'avg_duration', 'expected': 'lower'},
        'tag_fun': {'metric': 'completion_rate', 'expected': 'higher'},
    }
    
    results = {}
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (tag, expectation) in enumerate(tag_expectations.items()):
        metric = expectation['metric']
        expected_dir = expectation['expected']
        
        if tag not in df.columns or metric not in df.columns:
            continue
        
        # Split by tag presence
        has_tag = df[df[tag] > 0][metric].dropna()
        no_tag = df[df[tag] == 0][metric].dropna()
        
        if len(has_tag) < 3 or len(no_tag) < 3:
            print(f"\n{tag}: Insufficient data (has_tag={len(has_tag)}, no_tag={len(no_tag)})")
            continue
        
        # Statistical test
        stat, p = stats.mannwhitneyu(has_tag, no_tag, alternative='two-sided')
        effect_size = (has_tag.mean() - no_tag.mean()) / (df[metric].std() + 0.001)
        
        # Check if direction matches expectation
        actual_higher = has_tag.mean() > no_tag.mean()
        matches_expectation = (actual_higher and expected_dir == 'higher') or \
                              (not actual_higher and expected_dir == 'lower')
        
        results[tag] = {
            'metric': metric,
            'has_tag_mean': has_tag.mean(),
            'no_tag_mean': no_tag.mean(),
            'u_statistic': stat,
            'p_value': p,
            'effect_size': effect_size,
            'matches_expectation': matches_expectation
        }
        
        print(f"\n{tag} → {metric}:")
        print(f"  Has tag (n={len(has_tag)}): mean = {has_tag.mean():.3f}")
        print(f"  No tag (n={len(no_tag)}): mean = {no_tag.mean():.3f}")
        print(f"  Mann-Whitney U: {stat:.1f}, p = {p:.4f}")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
        print(f"  Matches expectation ({expected_dir}): {matches_expectation}")
        
        # Plot
        ax = axes[i]
        data_to_plot = [no_tag.values, has_tag.values]
        bp = ax.boxplot(data_to_plot, labels=['No Tag', 'Has Tag'])
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{tag} → {metric}\n(p={p:.3f})', fontsize=12)
        
        # Color based on significance
        if p < 0.05:
            ax.set_facecolor('#e6ffe6' if matches_expectation else '#ffe6e6')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h6_tag_telemetry.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results

def h7_tag_feature_importance(level_stats):
    """
    H7: Which features best predict each tag?
    
    Use correlation analysis as a simpler alternative to logistic regression.
    """
    print("\n" + "="*60)
    print("H7: Tag Feature Importance")
    print("="*60)
    
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    # Define features and tags
    feature_cols = ['avg_deaths', 'avg_duration', 'avg_jumps', 'completion_rate']
    tag_cols = ['tag_fun', 'tag_boring', 'tag_too_hard', 'tag_too_easy', 
                'tag_creative', 'tag_good_flow', 'tag_unfair']
    
    # Check available columns
    available_features = [c for c in feature_cols if c in df.columns]
    available_tags = [c for c in tag_cols if c in df.columns]
    
    # Binary tag presence
    for tag in available_tags:
        df[f'{tag}_binary'] = (df[tag] > 0).astype(int)
    
    # Compute point-biserial correlations
    correlations = []
    
    for tag in available_tags:
        tag_binary = f'{tag}_binary'
        for feature in available_features:
            valid = df[[tag_binary, feature]].dropna()
            if len(valid) > 5:
                corr, p = stats.pointbiserialr(valid[tag_binary], valid[feature])
                correlations.append({
                    'tag': tag,
                    'feature': feature,
                    'correlation': corr,
                    'p_value': p,
                    'abs_corr': abs(corr)
                })
    
    if not correlations:
        print("⚠️  Insufficient data for correlation analysis")
        return {}
    
    corr_df = pd.DataFrame(correlations)
    
    # Pivot for heatmap
    corr_pivot = corr_df.pivot(index='tag', columns='feature', values='correlation')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Correlation heatmap
    ax1 = axes[0]
    sns.heatmap(corr_pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax1,
                vmin=-0.5, vmax=0.5)
    ax1.set_title('H7: Tag-Feature Correlations', fontsize=14)
    
    # 2. Top correlations bar chart
    ax2 = axes[1]
    top_corr = corr_df.nlargest(10, 'abs_corr')
    colors = ['green' if c > 0 else 'red' for c in top_corr['correlation']]
    bars = ax2.barh(top_corr['tag'] + '\n→ ' + top_corr['feature'], 
                    top_corr['correlation'], color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Correlation', fontsize=12)
    ax2.set_title('Top 10 Tag-Feature Correlations', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h7_tag_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print results
    print("\nTop correlations (|r| > 0.15):")
    significant_corr = corr_df[corr_df['abs_corr'] > 0.15].sort_values('abs_corr', ascending=False)
    for _, row in significant_corr.iterrows():
        sig = '*' if row['p_value'] < 0.05 else ''
        print(f"  {row['tag']} ↔ {row['feature']}: r = {row['correlation']:.3f} (p={row['p_value']:.3f}){sig}")
    
    return corr_df.to_dict('records')

def tag_distribution_analysis(level_stats):
    """Analyze overall tag usage patterns."""
    print("\n" + "="*60)
    print("Tag Distribution Analysis")
    print("="*60)
    
    df = level_stats.copy()
    
    # Tag columns
    tag_cols = ['tag_fun', 'tag_boring', 'tag_too_hard', 'tag_too_easy', 
                'tag_creative', 'tag_good_flow', 'tag_unfair']
    available_tags = [c for c in tag_cols if c in df.columns]
    
    # Total tag counts
    tag_totals = df[available_tags].sum()
    
    # Levels with each tag
    tag_presence = (df[available_tags] > 0).sum()
    
    print("\nTag Usage Summary:")
    print("-" * 50)
    print(f"{'Tag':<20} {'Total Uses':<15} {'Levels with Tag':<15}")
    print("-" * 50)
    for tag in available_tags:
        print(f"{tag:<20} {int(tag_totals[tag]):<15} {int(tag_presence[tag]):<15}")
    
    # Plot tag distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Total tag usage
    ax1 = axes[0]
    tag_totals.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Tag', fontsize=12)
    ax1.set_ylabel('Total Uses', fontsize=12)
    ax1.set_title('Total Tag Usage Across All Levels', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 2. Tag co-occurrence
    ax2 = axes[1]
    tag_binary = (df[available_tags] > 0).astype(int)
    co_occurrence = tag_binary.T.dot(tag_binary)
    np.fill_diagonal(co_occurrence.values, 0)  # Remove self-co-occurrence
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Tag Co-occurrence Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tag_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Tag correlation
    print("\nTag Correlations (Phi coefficients):")
    tag_corr = tag_binary.corr()
    # Find strongest correlations
    for i, tag1 in enumerate(available_tags):
        for j, tag2 in enumerate(available_tags):
            if i < j and abs(tag_corr.loc[tag1, tag2]) > 0.2:
                print(f"  {tag1} ↔ {tag2}: φ = {tag_corr.loc[tag1, tag2]:.3f}")

def tag_by_generator(level_stats):
    """Analyze tag patterns by generator."""
    print("\n" + "="*60)
    print("Tag Patterns by Generator")
    print("="*60)
    
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    tag_cols = ['tag_fun', 'tag_boring', 'tag_too_hard', 'tag_too_easy', 
                'tag_creative', 'tag_good_flow', 'tag_unfair']
    available_tags = [c for c in tag_cols if c in df.columns]
    
    # Normalize tags by times shown
    for tag in available_tags:
        df[f'{tag}_rate'] = df[tag] / df['times_shown']
    
    rate_cols = [f'{tag}_rate' for tag in available_tags]
    
    # Group by generator
    gen_tags = df.groupby('generator_id')[rate_cols].mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(gen_tags, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('Tag Rate by Generator\n(proportion of plays with tag)', fontsize=14)
    ax.set_xlabel('Tag', fontsize=12)
    ax.set_ylabel('Generator', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tag_by_generator.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print extreme values
    print("\nGenerators with highest tag rates:")
    for col in rate_cols:
        top_gen = gen_tags[col].idxmax()
        top_val = gen_tags[col].max()
        if top_val > 0.1:
            print(f"  {col}: {top_gen} ({top_val:.2f})")

def main():
    """Run all RQ3 experiments."""
    print("="*60)
    print("RQ3: What Do Tags Actually Mean?")
    print("="*60)
    
    # Load data
    level_stats, telemetry, votes_df = load_data()
    
    print(f"\nData loaded:")
    print(f"  Levels: {len(level_stats)}")
    print(f"  Telemetry records: {len(telemetry)}")
    print(f"  Votes: {len(votes_df)}")
    
    # Run hypothesis tests
    h6_results = h6_tag_telemetry_correspondence(level_stats, telemetry)
    h7_results = h7_tag_feature_importance(level_stats)
    
    # Additional analysis
    tag_distribution_analysis(level_stats)
    tag_by_generator(level_stats)
    
    print("\n" + "="*60)
    print("RQ3 Analysis Complete")
    print("="*60)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    
    return {
        'h6': h6_results,
        'h7': h7_results,
    }

if __name__ == "__main__":
    results = main()
