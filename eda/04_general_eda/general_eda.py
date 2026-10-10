"""
PCG Arena EDA - General Exploratory Data Analysis
==================================================
Additional analyses:
- Global distributions
- Generator fingerprinting
- Confusion matrix / pairwise win rates
- Trajectory visualization
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
    """Load all prepared data."""
    level_stats = pd.read_csv(DATA_DIR / "level_stats_clean.csv")
    telemetry = pd.read_csv(DATA_DIR / "telemetry_flat.csv")
    trajectory_features = pd.read_csv(DATA_DIR / "trajectory_features.csv")
    generator_stats = pd.read_csv(DATA_DIR / "generator_stats.csv")
    
    # Load raw data
    with open(RAW_DATA_DIR / "pcg-arena-votes-2026-01-28.json", 'r') as f:
        votes_raw = json.load(f)
    votes_df = pd.DataFrame(votes_raw)
    
    with open(RAW_DATA_DIR / "pcg-arena-trajectories-2026-01-28.json", 'r') as f:
        trajectories_data = json.load(f)
    trajectories_raw = trajectories_data.get('data', []) if isinstance(trajectories_data, dict) else trajectories_data
    
    return level_stats, telemetry, trajectory_features, generator_stats, votes_df, trajectories_raw

def global_distributions(level_stats, telemetry, generator_stats):
    """Visualize global distributions of key metrics."""
    print("\n" + "="*60)
    print("Global Distributions")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Win rate distribution
    ax1 = axes[0, 0]
    level_stats['win_rate'].hist(bins=20, ax=ax1, edgecolor='black', alpha=0.7)
    ax1.axvline(level_stats['win_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {level_stats["win_rate"].mean():.2f}')
    ax1.set_xlabel('Win Rate', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Level Win Rates', fontsize=14)
    ax1.legend()
    
    # 2. Death rate distribution
    ax2 = axes[0, 1]
    deaths = level_stats['avg_deaths'].dropna()
    deaths[deaths <= 5].hist(bins=20, ax=ax2, edgecolor='black', alpha=0.7)
    ax2.axvline(deaths.mean(), color='red', linestyle='--', 
                label=f'Mean: {deaths.mean():.2f}')
    ax2.set_xlabel('Average Deaths per Play', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Death Rates', fontsize=14)
    ax2.legend()
    
    # 3. Completion rate distribution
    ax3 = axes[0, 2]
    level_stats['completion_rate'].hist(bins=20, ax=ax3, edgecolor='black', alpha=0.7)
    ax3.axvline(level_stats['completion_rate'].mean(), color='red', linestyle='--', 
                label=f'Mean: {level_stats["completion_rate"].mean():.2f}')
    ax3.set_xlabel('Completion Rate', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Completion Rates', fontsize=14)
    ax3.legend()
    
    # 4. Play duration distribution
    ax4 = axes[1, 0]
    durations = telemetry['duration_seconds'].dropna()
    durations[durations <= 60].hist(bins=30, ax=ax4, edgecolor='black', alpha=0.7)
    ax4.axvline(durations.mean(), color='red', linestyle='--', 
                label=f'Mean: {durations.mean():.1f}s')
    ax4.set_xlabel('Play Duration (seconds)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Play Durations', fontsize=14)
    ax4.legend()
    
    # 5. Jumps per play distribution
    ax5 = axes[1, 1]
    if 'jumps' in telemetry.columns:
        jumps = telemetry['jumps'].dropna()
        jumps[jumps <= 50].hist(bins=25, ax=ax5, edgecolor='black', alpha=0.7)
        ax5.axvline(jumps.mean(), color='red', linestyle='--', 
                    label=f'Mean: {jumps.mean():.1f}')
        ax5.set_xlabel('Jumps per Play', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Distribution of Jump Counts', fontsize=14)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Jump data not available', ha='center', va='center')
        ax5.set_title('Distribution of Jump Counts', fontsize=14)
    
    # 6. Levels per generator
    ax6 = axes[1, 2]
    gen_counts = level_stats['generator_id'].value_counts()
    ax6.bar(range(len(gen_counts)), gen_counts.values)
    ax6.set_xticks(range(len(gen_counts)))
    ax6.set_xticklabels(gen_counts.index, rotation=45, ha='right', fontsize=8)
    ax6.set_xlabel('Generator', fontsize=12)
    ax6.set_ylabel('Number of Levels', fontsize=12)
    ax6.set_title('Levels per Generator', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "global_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total levels: {len(level_stats)}")
    print(f"  Generators: {level_stats['generator_id'].nunique()}")
    print(f"  Mean win rate: {level_stats['win_rate'].mean():.2f}")
    print(f"  Mean death rate: {level_stats['avg_deaths'].mean():.2f}")
    print(f"  Mean completion rate: {level_stats['completion_rate'].mean():.2f}")
    print(f"  Mean play duration: {telemetry['duration_seconds'].mean():.1f}s")

def generator_fingerprinting(level_stats, generator_stats):
    """Create generator fingerprints based on multiple metrics."""
    print("\n" + "="*60)
    print("Generator Fingerprinting")
    print("="*60)
    
    # Metrics for fingerprinting
    metrics = ['win_rate', 'avg_deaths', 'completion_rate', 'difficulty_score']
    available_metrics = [m for m in metrics if m in generator_stats.columns]
    
    # Normalize metrics for radar chart
    df = generator_stats.copy()
    for metric in available_metrics:
        df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min() + 0.001)
    
    norm_metrics = [f'{m}_norm' for m in available_metrics]
    
    # Select top and bottom generators by win rate
    df_sorted = df.sort_values('win_rate', ascending=False)
    top_generators = df_sorted.head(4)['generator_id'].tolist()
    bottom_generators = df_sorted.tail(4)['generator_id'].tolist()
    
    # Plot radar chart for selected generators
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Top generators
    ax1 = axes[0]
    for gen in top_generators:
        gen_data = df[df['generator_id'] == gen]
        if len(gen_data) > 0:
            values = gen_data[norm_metrics].values.flatten().tolist()
            values += values[:1]
            ax1.plot(angles, values, 'o-', linewidth=2, label=gen[:12])
            ax1.fill(angles, values, alpha=0.15)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(available_metrics)
    ax1.set_title('Top Generators', fontsize=14)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Bottom generators
    ax2 = axes[1]
    for gen in bottom_generators:
        gen_data = df[df['generator_id'] == gen]
        if len(gen_data) > 0:
            values = gen_data[norm_metrics].values.flatten().tolist()
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=gen[:12])
            ax2.fill(angles, values, alpha=0.15)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(available_metrics)
    ax2.set_title('Bottom Generators', fontsize=14)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generator_fingerprints.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print fingerprint summary
    print("\nGenerator Fingerprints (normalized 0-1):")
    print(df[['generator_id'] + norm_metrics].round(2).to_string())

def pairwise_win_matrix(votes_df):
    """Create pairwise win rate matrix between generators."""
    print("\n" + "="*60)
    print("Pairwise Win Rate Matrix")
    print("="*60)
    
    # Get votes with both generators
    if 'winner_generator_id' not in votes_df.columns or 'loser_generator_id' not in votes_df.columns:
        print("⚠️  Missing winner/loser generator columns")
        return
    
    # Count head-to-head results
    generators = set(votes_df['winner_generator_id'].dropna()) | set(votes_df['loser_generator_id'].dropna())
    generators = sorted(generators)
    
    # Create win matrix
    win_matrix = pd.DataFrame(0.0, index=generators, columns=generators)
    count_matrix = pd.DataFrame(0, index=generators, columns=generators)
    
    for _, row in votes_df.iterrows():
        winner = row.get('winner_generator_id')
        loser = row.get('loser_generator_id')
        
        if pd.notna(winner) and pd.notna(loser) and winner in generators and loser in generators:
            count_matrix.loc[winner, loser] += 1
            count_matrix.loc[loser, winner] += 1
    
    # Calculate win rates
    for i in generators:
        for j in generators:
            total = count_matrix.loc[i, j]
            if total > 0:
                # i vs j: count how many times i beat j
                wins_i = votes_df[(votes_df['winner_generator_id'] == i) & 
                                  (votes_df['loser_generator_id'] == j)].shape[0]
                win_matrix.loc[i, j] = wins_i / total
    
    # Set diagonal to 0.5 (neutral)
    np.fill_diagonal(win_matrix.values, 0.5)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(win_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5, ax=ax,
                vmin=0, vmax=1, xticklabels=True, yticklabels=True)
    ax.set_xlabel('Opponent Generator', fontsize=12)
    ax.set_ylabel('Generator', fontsize=12)
    ax.set_title('Pairwise Win Rate Matrix\n(row beats column)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pairwise_win_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print notable matchups
    print("\nNotable head-to-head matchups:")
    for i in generators:
        for j in generators:
            if i < j and count_matrix.loc[i, j] >= 5:
                wr = win_matrix.loc[i, j]
                if wr >= 0.7 or wr <= 0.3:
                    print(f"  {i[:15]} vs {j[:15]}: {wr:.0%} (n={int(count_matrix.loc[i, j])})")

def trajectory_visualization(trajectories_raw, trajectory_features):
    """Visualize player trajectory patterns."""
    print("\n" + "="*60)
    print("Trajectory Visualization")
    print("="*60)
    
    if not trajectories_raw:
        print("⚠️  No trajectory data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Sample trajectories
    ax1 = axes[0, 0]
    sample_count = min(10, len(trajectories_raw))
    
    for i, traj in enumerate(trajectories_raw[:sample_count]):
        if 'positions' in traj and traj['positions']:
            positions = traj['positions']
            x_coords = [p.get('x', 0) for p in positions]
            y_coords = [p.get('y', 0) for p in positions]
            ax1.plot(x_coords, y_coords, alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title(f'Sample Player Trajectories (n={sample_count})', fontsize=14)
    ax1.invert_yaxis()  # Game coordinates often have y increasing downward
    
    # 2. Spatial coverage distribution
    ax2 = axes[0, 1]
    if 'unique_tiles_visited' in trajectory_features.columns:
        trajectory_features['unique_tiles_visited'].hist(bins=20, ax=ax2, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Unique Tiles Visited', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Spatial Coverage Distribution', fontsize=14)
    
    # 3. Progress (max X) distribution
    ax3 = axes[1, 0]
    if 'max_x_reached' in trajectory_features.columns:
        trajectory_features['max_x_reached'].hist(bins=20, ax=ax3, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Max X Position Reached', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Level Progress Distribution', fontsize=14)
    
    # 4. Speed vs coverage scatter
    ax4 = axes[1, 1]
    if 'avg_speed' in trajectory_features.columns and 'unique_tiles_visited' in trajectory_features.columns:
        ax4.scatter(trajectory_features['avg_speed'], trajectory_features['unique_tiles_visited'], 
                   alpha=0.5, s=30)
        ax4.set_xlabel('Average Speed', fontsize=12)
        ax4.set_ylabel('Unique Tiles Visited', fontsize=12)
        ax4.set_title('Speed vs Exploration', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trajectory_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print trajectory stats
    print(f"\nTrajectory Statistics:")
    print(f"  Total trajectories: {len(trajectories_raw)}")
    if len(trajectory_features) > 0:
        print(f"  Avg unique tiles: {trajectory_features['unique_tiles_visited'].mean():.1f}")
        print(f"  Avg max X reached: {trajectory_features['max_x_reached'].mean():.1f}")
        print(f"  Avg speed: {trajectory_features['avg_speed'].mean():.2f}")

def correlation_analysis(level_stats):
    """Comprehensive correlation analysis."""
    print("\n" + "="*60)
    print("Correlation Analysis")
    print("="*60)
    
    # Select numeric columns
    numeric_cols = ['win_rate', 'avg_deaths', 'completion_rate', 'avg_duration',
                    'avg_jumps', 'times_shown', 'difficulty_score']
    available_cols = [c for c in numeric_cols if c in level_stats.columns]
    
    # Compute correlation matrix
    corr_matrix = level_stats[available_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix of Level Metrics', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print strong correlations
    print("\nStrong correlations (|r| > 0.5):")
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            if i < j:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.5:
                    print(f"  {col1} ↔ {col2}: r = {corr:.3f}")

def main():
    """Run all general EDA analyses."""
    print("="*60)
    print("General Exploratory Data Analysis")
    print("="*60)
    
    # Load data
    level_stats, telemetry, trajectory_features, generator_stats, votes_df, trajectories_raw = load_data()
    
    print(f"\nData loaded:")
    print(f"  Levels: {len(level_stats)}")
    print(f"  Telemetry records: {len(telemetry)}")
    print(f"  Trajectories: {len(trajectories_raw)}")
    print(f"  Votes: {len(votes_df)}")
    print(f"  Generators: {len(generator_stats)}")
    
    # Run analyses
    global_distributions(level_stats, telemetry, generator_stats)
    generator_fingerprinting(level_stats, generator_stats)
    pairwise_win_matrix(votes_df)
    trajectory_visualization(trajectories_raw, trajectory_features)
    correlation_analysis(level_stats)
    
    print("\n" + "="*60)
    print("General EDA Complete")
    print("="*60)
    print(f"\nPlots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
