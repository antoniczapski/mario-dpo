"""
PCG Arena EDA - RQ1: What Makes a Good Generator/Level?
=========================================================
Hypotheses tested:
- H1: Optimized Difficulty (Flow Channel) - moderate difficulty → higher win rates
- H2: Structural Variety - terrain variance → preference
- H3: Path Freedom (Agency) - exploration → preference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Paths
EDA_DIR = Path(__file__).parent.parent
DATA_DIR = EDA_DIR / "00_data_preparation"
PLOTS_DIR = EDA_DIR / "plots"
OUTPUT_DIR = EDA_DIR / "01_rq1_good_generators"

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data():
    """Load prepared data."""
    level_stats = pd.read_csv(DATA_DIR / "level_stats_clean.csv")
    telemetry = pd.read_csv(DATA_DIR / "telemetry_flat.csv")
    trajectory_features = pd.read_csv(DATA_DIR / "trajectory_features.csv")
    generator_stats = pd.read_csv(DATA_DIR / "generator_stats.csv")
    return level_stats, telemetry, trajectory_features, generator_stats

def h1_difficulty_vs_winrate(level_stats, telemetry):
    """
    H1: Levels with moderate difficulty (15-40% death rate) have higher win rates
    than levels with 0% or >60% death rates.
    
    Test: Plot Win Rate vs Death Rate, check for inverted U-shape.
    """
    print("\n" + "="*60)
    print("H1: Optimized Difficulty (Flow Channel)")
    print("="*60)
    
    # Filter levels with enough data
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    # Calculate death rate (deaths per play)
    df['death_rate'] = df['avg_deaths']
    
    # Remove extreme outliers
    df = df[df['death_rate'] <= 10]
    
    # Create difficulty bins
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]
    labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
    df['difficulty_bin'] = pd.cut(df['death_rate'], bins=bins, labels=labels)
    
    # Plot 1: Scatter plot of win rate vs death rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with regression
    ax1 = axes[0]
    ax1.scatter(df['death_rate'], df['win_rate'], alpha=0.5, s=30)
    
    # Add polynomial regression
    valid = df[['death_rate', 'win_rate']].dropna()
    if len(valid) > 10:
        z = np.polyfit(valid['death_rate'], valid['win_rate'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(valid['death_rate'].min(), valid['death_rate'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Quadratic fit')
    
    ax1.set_xlabel('Average Deaths per Play', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('H1: Win Rate vs Difficulty (Deaths)', fontsize=14)
    ax1.legend()
    
    # Box plot by difficulty bins
    ax2 = axes[1]
    df_binned = df.dropna(subset=['difficulty_bin'])
    if len(df_binned) > 0:
        df_binned.boxplot(column='win_rate', by='difficulty_bin', ax=ax2)
        ax2.set_xlabel('Deaths per Play (binned)', fontsize=12)
        ax2.set_ylabel('Win Rate', fontsize=12)
        ax2.set_title('Win Rate by Difficulty Bin', fontsize=14)
        plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h1_difficulty_vs_winrate.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistical test: Correlation
    corr, p_value = stats.spearmanr(df['death_rate'].dropna(), df['win_rate'].dropna())
    
    # Check for inverted-U by binned analysis
    binned_means = df.groupby('difficulty_bin')['win_rate'].mean()
    
    print(f"\nSample size: {len(df)} levels (with ≥3 plays)")
    print(f"\nSpearman correlation (death_rate vs win_rate): r = {corr:.3f}, p = {p_value:.4f}")
    print(f"\nWin rate by difficulty bin:")
    print(binned_means.to_string())
    
    # Check for peak in middle bins
    if len(binned_means) >= 3:
        mid_idx = len(binned_means) // 2
        peak_in_middle = binned_means.iloc[mid_idx] > binned_means.iloc[0] and \
                         binned_means.iloc[mid_idx] > binned_means.iloc[-1]
        print(f"\nInverted-U pattern (peak in middle): {peak_in_middle}")
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'binned_means': binned_means.to_dict(),
        'sample_size': len(df)
    }

def h2_structural_variety(level_stats, telemetry):
    """
    H2: Levels with higher variance in terrain height and more distinct enemy 
    types are preferred over flat or repetitive levels.
    
    Note: Structural features are null in current data, so we use tag-based proxies.
    """
    print("\n" + "="*60)
    print("H2: Structural Variety")
    print("="*60)
    
    # Since structural features are null, analyze tag patterns as proxy
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    # Create "variety score" based on tags (creative = varied, boring = repetitive)
    df['creative_rate'] = df['tag_creative'] / df['times_shown']
    df['boring_rate'] = df['tag_boring'] / df['times_shown']
    df['variety_proxy'] = df['creative_rate'] - df['boring_rate']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Creative rate vs win rate
    ax1 = axes[0]
    valid = df[df['creative_rate'] > 0]
    ax1.scatter(df['creative_rate'], df['win_rate'], alpha=0.5, s=30)
    ax1.set_xlabel('Creative Tag Rate', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('H2: Win Rate vs "Creative" Tag Rate', fontsize=14)
    
    if len(valid) > 5:
        corr_creative, p_creative = stats.spearmanr(
            df['creative_rate'].fillna(0), 
            df['win_rate'].fillna(0)
        )
        ax1.annotate(f'r = {corr_creative:.3f}\np = {p_creative:.4f}', 
                     xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10)
    
    # Boring rate vs win rate
    ax2 = axes[1]
    ax2.scatter(df['boring_rate'], df['win_rate'], alpha=0.5, s=30, color='orange')
    ax2.set_xlabel('Boring Tag Rate', fontsize=12)
    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.set_title('H2: Win Rate vs "Boring" Tag Rate', fontsize=14)
    
    corr_boring, p_boring = stats.spearmanr(
        df['boring_rate'].fillna(0), 
        df['win_rate'].fillna(0)
    )
    ax2.annotate(f'r = {corr_boring:.3f}\np = {p_boring:.4f}', 
                 xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h2_structural_variety.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSample size: {len(df)} levels")
    print(f"Levels with 'creative' tag: {(df['tag_creative'] > 0).sum()}")
    print(f"Levels with 'boring' tag: {(df['tag_boring'] > 0).sum()}")
    print(f"\nCorrelation (creative_rate vs win_rate): r = {corr_creative:.3f}, p = {p_creative:.4f}")
    print(f"Correlation (boring_rate vs win_rate): r = {corr_boring:.3f}, p = {p_boring:.4f}")
    
    # Note about missing data
    print("\n⚠️  NOTE: Structural features (enemy_density, gap_count, etc.) are null")
    print("   in the current export. Using tag-based proxies instead.")
    
    return {
        'corr_creative': corr_creative,
        'p_creative': p_creative,
        'corr_boring': corr_boring,
        'p_boring': p_boring,
    }

def h3_path_freedom(trajectory_features, telemetry):
    """
    H3: Players prefer levels that allow for more exploration (higher spatial 
    coverage in trajectories) rather than strict linear paths.
    """
    print("\n" + "="*60)
    print("H3: Path Freedom (Agency)")
    print("="*60)
    
    # Merge trajectory features with telemetry outcomes
    df = trajectory_features.merge(
        telemetry[['vote_id', 'side', 'level_id', 'won', 'generator_id']],
        on=['vote_id', 'level_id', 'side'],
        how='left'
    )
    
    df = df.dropna(subset=['won'])
    
    if len(df) < 10:
        print("⚠️  Insufficient trajectory data for analysis")
        return {}
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Spatial coverage vs win outcome
    ax1 = axes[0, 0]
    df.boxplot(column='unique_tiles_visited', by='won', ax=ax1)
    ax1.set_xlabel('Won Battle', fontsize=12)
    ax1.set_ylabel('Unique Tiles Visited', fontsize=12)
    ax1.set_title('Spatial Coverage by Outcome', fontsize=14)
    plt.suptitle('')
    
    # 2. Backtrack ratio vs win outcome
    ax2 = axes[0, 1]
    df.boxplot(column='backtrack_ratio', by='won', ax=ax2)
    ax2.set_xlabel('Won Battle', fontsize=12)
    ax2.set_ylabel('Backtrack Ratio', fontsize=12)
    ax2.set_title('Exploration (Backtracking) by Outcome', fontsize=14)
    
    # 3. Max X reached vs win
    ax3 = axes[1, 0]
    df.boxplot(column='max_x_reached', by='won', ax=ax3)
    ax3.set_xlabel('Won Battle', fontsize=12)
    ax3.set_ylabel('Max X Position Reached', fontsize=12)
    ax3.set_title('Progress by Outcome', fontsize=14)
    
    # 4. Average speed vs win
    ax4 = axes[1, 1]
    df.boxplot(column='avg_speed', by='won', ax=ax4)
    ax4.set_xlabel('Won Battle', fontsize=12)
    ax4.set_ylabel('Average Speed', fontsize=12)
    ax4.set_title('Movement Speed by Outcome', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3_path_freedom.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistical tests
    won = df[df['won'] == True]
    lost = df[df['won'] == False]
    
    results = {}
    
    for metric in ['unique_tiles_visited', 'backtrack_ratio', 'max_x_reached', 'avg_speed']:
        won_vals = won[metric].dropna()
        lost_vals = lost[metric].dropna()
        
        if len(won_vals) > 2 and len(lost_vals) > 2:
            stat, p = stats.mannwhitneyu(won_vals, lost_vals, alternative='two-sided')
            results[metric] = {
                'won_mean': won_vals.mean(),
                'lost_mean': lost_vals.mean(),
                'u_statistic': stat,
                'p_value': p
            }
            print(f"\n{metric}:")
            print(f"  Won mean: {won_vals.mean():.2f}, Lost mean: {lost_vals.mean():.2f}")
            print(f"  Mann-Whitney U: {stat:.1f}, p = {p:.4f}")
    
    return results

def generator_analysis(generator_stats, level_stats):
    """Analyze what distinguishes top-performing generators."""
    print("\n" + "="*60)
    print("Generator-Level Analysis")
    print("="*60)
    
    df = generator_stats.copy()
    
    # Sort by win rate
    df = df.sort_values('win_rate', ascending=False)
    
    print("\nGenerator Rankings (by win rate):")
    print(df[['generator_id', 'num_levels', 'times_shown', 'win_rate', 
              'avg_deaths', 'completion_rate']].to_string())
    
    # Plot: Generator comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Win rate by generator
    ax1 = axes[0, 0]
    bars = ax1.barh(df['generator_id'], df['win_rate'])
    ax1.set_xlabel('Win Rate', fontsize=12)
    ax1.set_ylabel('Generator', fontsize=12)
    ax1.set_title('Win Rate by Generator', fontsize=14)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    # 2. Difficulty vs Win Rate
    ax2 = axes[0, 1]
    ax2.scatter(df['avg_deaths'], df['win_rate'], s=df['times_shown']/5, alpha=0.7)
    for _, row in df.iterrows():
        ax2.annotate(row['generator_id'][:8], 
                     (row['avg_deaths'], row['win_rate']),
                     fontsize=8, alpha=0.7)
    ax2.set_xlabel('Average Deaths', fontsize=12)
    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.set_title('Generator: Difficulty vs Win Rate\n(size = # plays)', fontsize=14)
    
    # 3. Tag profile heatmap
    ax3 = axes[1, 0]
    tag_cols = ['tag_fun', 'tag_boring', 'tag_too_hard', 'tag_too_easy', 
                'tag_creative', 'tag_good_flow', 'tag_unfair']
    tag_data = df.set_index('generator_id')[tag_cols]
    # Normalize by times shown
    tag_data_norm = tag_data.div(df.set_index('generator_id')['times_shown'], axis=0)
    sns.heatmap(tag_data_norm, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax3)
    ax3.set_title('Tag Rate by Generator (normalized)', fontsize=14)
    
    # 4. Completion rate vs win rate
    ax4 = axes[1, 1]
    ax4.scatter(df['completion_rate'], df['win_rate'], s=100, alpha=0.7)
    for _, row in df.iterrows():
        ax4.annotate(row['generator_id'][:8], 
                     (row['completion_rate'], row['win_rate']),
                     fontsize=8, alpha=0.7)
    ax4.set_xlabel('Completion Rate', fontsize=12)
    ax4.set_ylabel('Win Rate', fontsize=12)
    ax4.set_title('Completion Rate vs Win Rate', fontsize=14)
    
    # Add regression line
    valid = df[['completion_rate', 'win_rate']].dropna()
    if len(valid) > 3:
        z = np.polyfit(valid['completion_rate'], valid['win_rate'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['completion_rate'].min(), valid['completion_rate'].max(), 100)
        ax4.plot(x_line, p(x_line), 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generator_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlation analysis
    print("\n\nCorrelation Matrix (generators):")
    corr_cols = ['win_rate', 'avg_deaths', 'completion_rate', 'difficulty_score']
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.to_string())

def main():
    """Run all RQ1 experiments."""
    print("="*60)
    print("RQ1: What Makes a Good Generator/Level?")
    print("="*60)
    
    # Load data
    level_stats, telemetry, trajectory_features, generator_stats = load_data()
    
    # Run hypothesis tests
    h1_results = h1_difficulty_vs_winrate(level_stats, telemetry)
    h2_results = h2_structural_variety(level_stats, telemetry)
    h3_results = h3_path_freedom(trajectory_features, telemetry)
    
    # Generator-level analysis
    generator_analysis(generator_stats, level_stats)
    
    print("\n" + "="*60)
    print("RQ1 Analysis Complete")
    print("="*60)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    
    return {
        'h1': h1_results,
        'h2': h2_results,
        'h3': h3_results,
    }

if __name__ == "__main__":
    results = main()
