"""
PCG Arena EDA - RQ2: Are Players Consistent in Their Preferences?
==================================================================
Hypotheses tested:
- H4: Preference clusters exist based on player skill/playstyle
- H5: Higher-skill players show more consistent preferences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
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
    player_patterns = pd.read_csv(DATA_DIR / "player_voting_patterns.csv")
    
    # Load raw votes for additional analysis
    with open(RAW_DATA_DIR / "pcg-arena-votes-2026-01-28.json", 'r') as f:
        votes_raw = json.load(f)
    votes_df = pd.DataFrame(votes_raw)
    
    # Load player profiles
    with open(RAW_DATA_DIR / "pcg-arena-player-profiles-2026-01-28.json", 'r') as f:
        profiles_raw = json.load(f)
    profiles_df = pd.DataFrame(profiles_raw)
    
    return player_patterns, votes_df, profiles_df

def h4_preference_clusters(player_patterns):
    """
    H4: There exist distinct preference clusters (e.g., challenge-seekers vs. 
    flow-enjoyers) that can be identified by clustering on player voting vectors.
    
    Test: Hierarchical clustering on player vote/tag patterns.
    """
    print("\n" + "="*60)
    print("H4: Preference Clusters")
    print("="*60)
    
    # Filter players with enough data
    df = player_patterns[player_patterns['num_votes'] >= 5].copy()
    
    if len(df) < 6:
        print("⚠️  Insufficient players with ≥5 votes for clustering")
        return {}
    
    # Feature columns for clustering
    feature_cols = ['avg_deaths', 'avg_duration', 'completion_rate', 
                    'avg_jumps', 'preferred_difficulty']
    
    # Check which columns exist
    available_cols = [c for c in feature_cols if c in df.columns]
    
    if len(available_cols) < 3:
        print("⚠️  Insufficient features for clustering")
        return {}
    
    # Prepare feature matrix
    X = df[available_cols].fillna(0).values
    
    # Standardize manually
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 0.0001
    X_scaled = (X - X_mean) / X_std
    
    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='ward')
    optimal_k = 3 if len(df) >= 9 else 2
    df['cluster'] = fcluster(linkage_matrix, t=optimal_k, criterion='maxclust') - 1  # 0-indexed
    
    # Simple 2D projection using first two features
    df['dim1'] = X_scaled[:, 0]
    df['dim2'] = X_scaled[:, 1] if X_scaled.shape[1] > 1 else 0
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter with clusters
    ax1 = axes[0, 0]
    for cluster in range(optimal_k):
        mask = df['cluster'] == cluster
        ax1.scatter(df.loc[mask, 'dim1'], df.loc[mask, 'dim2'], 
                   label=f'Cluster {cluster}', s=100, alpha=0.7)
    ax1.set_xlabel(f'{available_cols[0]} (standardized)', fontsize=12)
    ax1.set_ylabel(f'{available_cols[1]} (standardized)', fontsize=12)
    ax1.set_title('H4: Player Preference Clusters', fontsize=14)
    ax1.legend()
    
    # 2. Dendrogram
    ax2 = axes[0, 1]
    dendrogram(linkage_matrix, ax=ax2, leaf_rotation=90, leaf_font_size=8)
    ax2.set_xlabel('Player Index', fontsize=12)
    ax2.set_ylabel('Distance', fontsize=12)
    ax2.set_title('Hierarchical Clustering Dendrogram', fontsize=14)
    
    # 3. Cluster profiles (bar chart)
    ax3 = axes[1, 0]
    cluster_means = df.groupby('cluster')[available_cols].mean()
    cluster_means.T.plot(kind='bar', ax=ax3)
    ax3.set_xlabel('Feature', fontsize=12)
    ax3.set_ylabel('Mean Value', fontsize=12)
    ax3.set_title('Cluster Profiles', fontsize=14)
    ax3.legend(title='Cluster')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Cluster sizes and vote counts
    ax4 = axes[1, 1]
    cluster_summary = df.groupby('cluster').agg({
        'player_id': 'count',
        'num_votes': 'sum'
    }).rename(columns={'player_id': 'num_players'})
    
    x = np.arange(optimal_k)
    width = 0.35
    ax4.bar(x - width/2, cluster_summary['num_players'], width, label='# Players')
    ax4.bar(x + width/2, cluster_summary['num_votes'], width, label='# Votes')
    ax4.set_xlabel('Cluster', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Cluster Sizes', fontsize=14)
    ax4.legend()
    ax4.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h4_preference_clusters.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print cluster profiles
    print(f"\nClustering on {len(df)} players with ≥5 votes")
    print(f"Features used: {available_cols}")
    print(f"Optimal K: {optimal_k}")
    print(f"\nCluster profiles:")
    print(cluster_means.round(3).to_string())
    
    print(f"\nCluster sizes:")
    print(cluster_summary.to_string())
    
    return {
        'n_clusters': optimal_k,
        'n_players': len(df),
        'cluster_sizes': cluster_summary['num_players'].to_dict(),
    }

def h5_skill_consistency(player_patterns, votes_df):
    """
    H5: Higher-skill players (measured by completion rate or avg progress) show 
    more consistent preferences than lower-skill players.
    
    Test: Compare coefficient of variation in votes across skill quartiles.
    """
    print("\n" + "="*60)
    print("H5: Skill-Consistency Relationship")
    print("="*60)
    
    # Filter players with enough votes
    df = player_patterns[player_patterns['num_votes'] >= 5].copy()
    
    if len(df) < 8:
        print("⚠️  Insufficient players for quartile analysis")
        return {}
    
    # Create skill quartiles based on completion rate
    df['skill_quartile'] = pd.qcut(df['completion_rate'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    
    # Measure consistency: lower coefficient of variation in play style = more consistent
    # Use available metrics for consistency
    metric_cols = ['avg_deaths', 'avg_duration', 'avg_jumps', 'preferred_difficulty']
    available_metrics = [c for c in metric_cols if c in df.columns]
    
    if len(available_metrics) >= 2:
        # CV of metrics (lower = more focused/consistent)
        df['metric_cv'] = df[available_metrics].std(axis=1) / (df[available_metrics].mean(axis=1) + 0.001)
    else:
        df['metric_cv'] = 0
    
    # Also use win rate consistency
    df['win_rate'] = df['num_wins'] / df['num_votes']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Box plot: Consistency (CV) by skill quartile
    ax1 = axes[0]
    df.boxplot(column='metric_cv', by='skill_quartile', ax=ax1)
    ax1.set_xlabel('Skill Quartile (by completion rate)', fontsize=12)
    ax1.set_ylabel('Metric Coefficient of Variation\n(lower = more consistent)', fontsize=12)
    ax1.set_title('H5: Play Style Consistency by Skill Level', fontsize=14)
    plt.suptitle('')
    
    # 2. Scatter: Skill vs Consistency
    ax2 = axes[1]
    ax2.scatter(df['completion_rate'], df['metric_cv'], alpha=0.6, s=60)
    ax2.set_xlabel('Completion Rate (Skill)', fontsize=12)
    ax2.set_ylabel('Metric CV (lower = more consistent)', fontsize=12)
    ax2.set_title('Skill vs Play Style Consistency', fontsize=14)
    
    # Add regression
    valid = df[['completion_rate', 'metric_cv']].dropna()
    if len(valid) > 3:
        corr, p = stats.spearmanr(valid['completion_rate'], valid['metric_cv'])
        ax2.annotate(f'r = {corr:.3f}\np = {p:.4f}', 
                     xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h5_skill_consistency.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistical tests
    print(f"\nSample: {len(df)} players with ≥5 votes")
    
    # ANOVA across quartiles
    groups = [group['metric_cv'].values for name, group in df.groupby('skill_quartile')]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f"\nANOVA (metric_cv across skill quartiles): F = {f_stat:.3f}, p = {p_anova:.4f}")
    
    # Correlation
    corr, p_corr = stats.spearmanr(df['completion_rate'].fillna(0), df['metric_cv'].fillna(0))
    print(f"Spearman correlation (skill vs consistency): r = {corr:.3f}, p = {p_corr:.4f}")
    
    # Quartile summary
    print("\nMetric CV by Skill Quartile:")
    quartile_summary = df.groupby('skill_quartile')['metric_cv'].agg(['mean', 'std', 'count'])
    print(quartile_summary.to_string())
    
    return {
        'correlation': corr,
        'p_value': p_corr,
        'quartile_means': df.groupby('skill_quartile')['metric_cv'].mean().to_dict()
    }

def voting_behavior_analysis(votes_df, profiles_df):
    """Additional analysis of voting behavior patterns."""
    print("\n" + "="*60)
    print("Voting Behavior Analysis")
    print("="*60)
    
    # Vote distribution by generator
    if 'winner_generator_id' in votes_df.columns:
        winner_counts = votes_df['winner_generator_id'].value_counts()
        print("\nVotes won by generator:")
        print(winner_counts.head(10).to_string())
    
    # Time-based patterns
    if 'created_at' in votes_df.columns:
        votes_df['created_at'] = pd.to_datetime(votes_df['created_at'])
        votes_df['hour'] = votes_df['created_at'].dt.hour
        votes_df['day_of_week'] = votes_df['created_at'].dt.dayofweek
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Votes by hour
        ax1 = axes[0]
        votes_df['hour'].value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Votes', fontsize=12)
        ax1.set_title('Voting Activity by Hour', fontsize=14)
        
        # Votes by day
        ax2 = axes[1]
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts = votes_df['day_of_week'].value_counts().sort_index()
        ax2.bar(day_names, [day_counts.get(i, 0) for i in range(7)])
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Number of Votes', fontsize=12)
        ax2.set_title('Voting Activity by Day', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "voting_time_patterns.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Player engagement distribution
    if 'player_id' in votes_df.columns:
        votes_per_player = votes_df['player_id'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(votes_per_player, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(votes_per_player.median(), color='red', linestyle='--', 
                   label=f'Median: {votes_per_player.median():.0f}')
        ax.axvline(votes_per_player.mean(), color='green', linestyle='--', 
                   label=f'Mean: {votes_per_player.mean():.1f}')
        ax.set_xlabel('Votes per Player', fontsize=12)
        ax.set_ylabel('Number of Players', fontsize=12)
        ax.set_title('Player Engagement Distribution', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "player_engagement_dist.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVotes per player statistics:")
        print(f"  Mean: {votes_per_player.mean():.1f}")
        print(f"  Median: {votes_per_player.median():.0f}")
        print(f"  Max: {votes_per_player.max()}")
        print(f"  Min: {votes_per_player.min()}")
    
    # Experienced players
    if len(profiles_df) > 0 and 'total_battles' in profiles_df.columns:
        print(f"\nPlayer profiles:")
        print(f"  Total players: {len(profiles_df)}")
        experienced = profiles_df[profiles_df['total_battles'] >= 10]
        print(f"  Players with ≥10 battles: {len(experienced)}")

def main():
    """Run all RQ2 experiments."""
    print("="*60)
    print("RQ2: Are Players Consistent in Their Preferences?")
    print("="*60)
    
    # Load data
    player_patterns, votes_df, profiles_df = load_data()
    
    print(f"\nData loaded:")
    print(f"  Players with voting data: {len(player_patterns)}")
    print(f"  Total votes: {len(votes_df)}")
    print(f"  Player profiles: {len(profiles_df)}")
    
    # Run hypothesis tests
    h4_results = h4_preference_clusters(player_patterns)
    h5_results = h5_skill_consistency(player_patterns, votes_df)
    
    # Additional behavior analysis
    voting_behavior_analysis(votes_df, profiles_df)
    
    print("\n" + "="*60)
    print("RQ2 Analysis Complete")
    print("="*60)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    
    return {
        'h4': h4_results,
        'h5': h5_results,
    }

if __name__ == "__main__":
    results = main()
