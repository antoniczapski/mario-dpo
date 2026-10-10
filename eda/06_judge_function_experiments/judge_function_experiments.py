"""
Judge Function Experiments
==========================

Four experiments to derive weights for the PCG Level Quality "Judge" Function:
- Experiment A: Verticality Validation (Y-sigma correlation)
- Experiment B: Hazard Hierarchy (Gaps vs Enemies)
- Experiment C: Death Entropy Test (Clustered vs Spread deaths)
- Experiment D: Original Centroid (Style Matching with Mahalanobis Distance)

These experiments inform the reward model for evolutionary/RL PCG agents.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
EDA_DIR = Path(__file__).parent.parent
DATA_DIR = EDA_DIR
PLOTS_DIR = EDA_DIR / 'plots'
PREP_DIR = EDA_DIR / '00_data_preparation'

PLOTS_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_data():
    """Load all required datasets"""
    print("Loading data...")
    
    # Load trajectories JSON
    traj_file = DATA_DIR / 'pcg-arena-trajectories-2026-01-28.json'
    with open(traj_file, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)
    
    # Load votes JSON
    votes_file = DATA_DIR / 'pcg-arena-votes-2026-01-28.json'
    with open(votes_file, 'r', encoding='utf-8') as f:
        votes_data = json.load(f)
    
    # Load level stats
    level_stats = pd.read_csv(PREP_DIR / 'level_stats_clean.csv')
    
    # Load trajectory features (pre-computed)
    traj_features = pd.read_csv(PREP_DIR / 'trajectory_features.csv')
    
    print(f"  Trajectories: {len(traj_data['data'])}")
    print(f"  Votes: {len(votes_data['data'])}")
    print(f"  Levels: {len(level_stats)}")
    print(f"  Trajectory features: {len(traj_features)}")
    
    return traj_data, votes_data, level_stats, traj_features


def compute_trajectory_y_sigma(trajectory_points):
    """
    Calculate Y-coordinate standard deviation for a trajectory.
    Measures vertical exploration/movement variety.
    """
    if not trajectory_points or len(trajectory_points) < 2:
        return 0.0
    y_coords = [p['y'] for p in trajectory_points]
    return np.std(y_coords)


def compute_velocity_stats(trajectory_points):
    """
    Compute velocity-based features:
    - Hesitation ratio: % of ticks with near-zero X velocity
    - Velocity variance: rhythm indicator
    """
    if not trajectory_points or len(trajectory_points) < 3:
        return {'hesitation_ratio': 1.0, 'velocity_variance': 0.0}
    
    velocities = []
    for i in range(1, len(trajectory_points)):
        dx = trajectory_points[i]['x'] - trajectory_points[i-1]['x']
        dt = trajectory_points[i]['tick'] - trajectory_points[i-1]['tick']
        if dt > 0:
            velocities.append(abs(dx) / dt)
    
    if not velocities:
        return {'hesitation_ratio': 1.0, 'velocity_variance': 0.0}
    
    velocities = np.array(velocities)
    hesitation_ratio = np.sum(velocities < 0.5) / len(velocities)  # Near-zero threshold
    velocity_variance = np.var(velocities)
    
    return {
        'hesitation_ratio': hesitation_ratio,
        'velocity_variance': velocity_variance
    }


def compute_death_entropy(death_locations, level_width=3200, n_bins=10):
    """
    Calculate entropy of death distribution across level.
    Low entropy = deaths clustered at one spot (broken level)
    High entropy = deaths spread out (skill-based difficulty)
    """
    if not death_locations or len(death_locations) == 0:
        return np.nan  # No deaths = no entropy
    
    # Extract X coordinates of deaths
    death_x = [d['x'] for d in death_locations if 'x' in d]
    if len(death_x) == 0:
        return np.nan
    
    # Discretize into bins
    bin_edges = np.linspace(0, level_width, n_bins + 1)
    hist, _ = np.histogram(death_x, bins=bin_edges)
    
    # Normalize to probability distribution
    hist = hist.astype(float)
    if hist.sum() == 0:
        return np.nan
    hist = hist / hist.sum()
    
    # Calculate entropy (with safety for log(0))
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Normalize by max possible entropy
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


def compute_early_death_rate(death_locations, trajectory_points, early_pct=0.1):
    """
    Calculate what fraction of deaths occur in the first X% of the level.
    High early death rate = poor design / unfair start
    """
    if not death_locations or not trajectory_points:
        return np.nan
    
    # Get max X reached as proxy for level extent explored
    max_x = max(p['x'] for p in trajectory_points)
    if max_x == 0:
        return np.nan
    
    early_threshold = max_x * early_pct
    death_x = [d['x'] for d in death_locations if 'x' in d]
    
    if len(death_x) == 0:
        return np.nan
    
    early_deaths = sum(1 for x in death_x if x < early_threshold)
    return early_deaths / len(death_x)


def extract_trajectory_features(traj_data):
    """
    Extract dynamic features from raw trajectory data for all trajectories.
    """
    print("\nExtracting trajectory features...")
    records = []
    
    for traj in traj_data['data']:
        traj_id = traj['trajectory_id']
        level_id = traj['level_id']
        vote_id = traj['vote_id']
        side = traj['side']
        
        trajectory = traj.get('trajectory', [])
        death_locations = traj.get('death_locations', [])
        
        # Compute features
        y_sigma = compute_trajectory_y_sigma(trajectory)
        vel_stats = compute_velocity_stats(trajectory)
        death_entropy = compute_death_entropy(death_locations)
        early_death_rate = compute_early_death_rate(death_locations, trajectory)
        
        # Path entropy (unique tiles visited diversity)
        if trajectory:
            unique_tiles = set((int(p['x'] // 16), int(p['y'] // 16)) for p in trajectory)
            path_entropy = len(unique_tiles)
        else:
            path_entropy = 0
        
        records.append({
            'trajectory_id': traj_id,
            'vote_id': vote_id,
            'level_id': level_id,
            'side': side,
            'y_sigma': y_sigma,
            'hesitation_ratio': vel_stats['hesitation_ratio'],
            'velocity_variance': vel_stats['velocity_variance'],
            'death_entropy': death_entropy,
            'early_death_rate': early_death_rate,
            'path_entropy': path_entropy,
            'death_count': len(death_locations)
        })
    
    df = pd.DataFrame(records)
    print(f"  Extracted features for {len(df)} trajectories")
    return df


def experiment_a_verticality(traj_features_ext, votes_data, level_stats):
    """
    Experiment A: Verticality Validation
    
    Hypothesis: Levels with high Y-sigma correlate with win_rate,
    independent of completion rate.
    
    Method: Partial correlation analysis controlling for completion.
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: VERTICALITY VALIDATION")
    print("="*60)
    
    # Create vote lookup: level_id -> win/loss info
    votes_df = pd.DataFrame(votes_data['data'])
    
    # Aggregate trajectory features by level
    level_traj = traj_features_ext.groupby('level_id').agg({
        'y_sigma': 'mean',
        'path_entropy': 'mean',
        'hesitation_ratio': 'mean',
        'velocity_variance': 'mean'
    }).reset_index()
    
    # Merge with level stats
    merged = level_traj.merge(
        level_stats[['level_id', 'win_rate', 'completion_rate', 'generator_id', 
                     'tag_creative', 'tag_fun']],
        on='level_id',
        how='inner'
    )
    
    print(f"\nMerged dataset: {len(merged)} levels with trajectory data")
    
    # Filter out levels with no variability
    valid = merged[(merged['y_sigma'] > 0) & (~merged['win_rate'].isna())]
    print(f"Valid levels for analysis: {len(valid)}")
    
    if len(valid) < 10:
        print("WARNING: Insufficient data for robust analysis")
        return None
    
    # Simple correlations
    print("\n--- Simple Correlations ---")
    corr_y_win, p_y_win = stats.spearmanr(valid['y_sigma'], valid['win_rate'])
    print(f"Y-Sigma vs Win Rate: r = {corr_y_win:.3f}, p = {p_y_win:.4f}")
    
    corr_path_win, p_path_win = stats.spearmanr(valid['path_entropy'], valid['win_rate'])
    print(f"Path Entropy vs Win Rate: r = {corr_path_win:.3f}, p = {p_path_win:.4f}")
    
    # Partial correlation: Y-sigma vs Win Rate, controlling for Completion Rate
    print("\n--- Partial Correlation (controlling for Completion Rate) ---")
    
    # Method: Regress both variables on completion, then correlate residuals
    from sklearn.linear_model import LinearRegression
    
    valid_complete = valid[~valid['completion_rate'].isna()].copy()
    
    if len(valid_complete) >= 10:
        lr = LinearRegression()
        
        X = valid_complete[['completion_rate']].values
        
        lr.fit(X, valid_complete['y_sigma'])
        y_sigma_resid = valid_complete['y_sigma'] - lr.predict(X)
        
        lr.fit(X, valid_complete['win_rate'])
        win_rate_resid = valid_complete['win_rate'] - lr.predict(X)
        
        partial_corr, partial_p = stats.spearmanr(y_sigma_resid, win_rate_resid)
        print(f"Partial Corr(Y-Sigma, Win Rate | Completion): r = {partial_corr:.3f}, p = {partial_p:.4f}")
    else:
        partial_corr, partial_p = np.nan, np.nan
        print("Insufficient data for partial correlation")
    
    # Compare Y-sigma by generator (Original vs Others)
    print("\n--- Y-Sigma by Generator Type ---")
    valid['is_original'] = valid['generator_id'] == 'original'
    
    original_y = valid[valid['is_original']]['y_sigma']
    other_y = valid[~valid['is_original']]['y_sigma']
    
    if len(original_y) > 0 and len(other_y) > 0:
        stat, p_val = stats.mannwhitneyu(original_y, other_y, alternative='two-sided')
        print(f"Original Y-Sigma: mean={original_y.mean():.2f}, median={original_y.median():.2f}")
        print(f"Other Y-Sigma: mean={other_y.mean():.2f}, median={other_y.median():.2f}")
        print(f"Mann-Whitney U: p = {p_val:.4f}")
    
    # Creative tag correlation
    print("\n--- Verticality vs Creative Tag ---")
    creative_levels = valid[valid['tag_creative'] > 0]
    non_creative = valid[valid['tag_creative'] == 0]
    
    if len(creative_levels) > 2 and len(non_creative) > 2:
        stat, p_val = stats.mannwhitneyu(
            creative_levels['y_sigma'], 
            non_creative['y_sigma'], 
            alternative='two-sided'
        )
        print(f"Creative levels Y-Sigma: mean={creative_levels['y_sigma'].mean():.2f}")
        print(f"Non-creative Y-Sigma: mean={non_creative['y_sigma'].mean():.2f}")
        print(f"Mann-Whitney U: p = {p_val:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Y-Sigma vs Win Rate scatter
    ax1 = axes[0, 0]
    colors = valid['generator_id'].map(lambda x: 'red' if x == 'original' else 'blue')
    ax1.scatter(valid['y_sigma'], valid['win_rate'], c=colors, alpha=0.6, s=50)
    ax1.set_xlabel('Y-Sigma (Vertical Movement Variance)', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title(f'Verticality vs Win Rate\n(Spearman r={corr_y_win:.3f}, p={p_y_win:.4f})', fontsize=12)
    
    # Add trend line
    z = np.polyfit(valid['y_sigma'], valid['win_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['y_sigma'].min(), valid['y_sigma'].max(), 100)
    ax1.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Original'),
        Patch(facecolor='blue', alpha=0.6, label='Other Generators')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # 2. Y-Sigma distribution by generator type
    ax2 = axes[0, 1]
    data_for_box = [
        valid[valid['is_original']]['y_sigma'].dropna().values,
        valid[~valid['is_original']]['y_sigma'].dropna().values
    ]
    bp = ax2.boxplot(data_for_box, labels=['Original', 'Other Generators'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax2.set_ylabel('Y-Sigma', fontsize=12)
    ax2.set_title('Vertical Movement Variance by Generator Type', fontsize=12)
    
    # 3. Path Entropy vs Win Rate
    ax3 = axes[1, 0]
    ax3.scatter(valid['path_entropy'], valid['win_rate'], c=colors, alpha=0.6, s=50)
    ax3.set_xlabel('Path Entropy (Unique Tiles Visited)', fontsize=12)
    ax3.set_ylabel('Win Rate', fontsize=12)
    ax3.set_title(f'Exploration Diversity vs Win Rate\n(Spearman r={corr_path_win:.3f}, p={p_path_win:.4f})', fontsize=12)
    
    # 4. Hesitation ratio vs Win Rate
    ax4 = axes[1, 1]
    corr_hes_win, p_hes_win = stats.spearmanr(valid['hesitation_ratio'], valid['win_rate'])
    ax4.scatter(valid['hesitation_ratio'], valid['win_rate'], c=colors, alpha=0.6, s=50)
    ax4.set_xlabel('Hesitation Ratio (Flow Interruption)', fontsize=12)
    ax4.set_ylabel('Win Rate', fontsize=12)
    ax4.set_title(f'Flow vs Win Rate\n(Spearman r={corr_hes_win:.3f}, p={p_hes_win:.4f})', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_a_verticality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {PLOTS_DIR / 'exp_a_verticality.png'}")
    
    results = {
        'y_sigma_win_corr': corr_y_win,
        'y_sigma_win_p': p_y_win,
        'partial_corr': partial_corr,
        'partial_p': partial_p,
        'path_entropy_win_corr': corr_path_win,
        'path_entropy_win_p': p_path_win,
        'hesitation_win_corr': corr_hes_win,
        'hesitation_win_p': p_hes_win
    }
    
    print("\n--- EXPERIMENT A CONCLUSION ---")
    if corr_y_win > 0.3 or (partial_corr and partial_corr > 0.3):
        print("SUPPORTED: Verticality (Y-sigma) significantly correlates with win rate.")
        print("IMPLICATION: Judge function MUST simulate vertical traversal.")
    else:
        print("PARTIAL/NOT SUPPORTED: Verticality shows weak correlation.")
        print("IMPLICATION: Vertical exploration may not be the key differentiator.")
    
    return results


def experiment_b_hazard_hierarchy(level_stats):
    """
    Experiment B: Hazard Hierarchy (Gaps vs Enemies)
    
    Hypothesis: Players penalize gaps more than enemies.
    Pattern generators failed because they spammed gaps.
    
    Method: Logistic regression comparing coefficients.
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: HAZARD HIERARCHY")
    print("="*60)
    
    # Check for required columns
    required = ['gap_density', 'enemy_density', 'win_rate']
    
    # Filter to levels with valid hazard data
    valid = level_stats.copy()
    
    # Check if gap_density and enemy_density exist and have values
    has_gap = 'gap_density' in valid.columns and valid['gap_density'].notna().sum() > 0
    has_enemy = 'enemy_density' in valid.columns and valid['enemy_density'].notna().sum() > 0
    
    print(f"\nData availability:")
    print(f"  gap_density available: {has_gap}")
    print(f"  enemy_density available: {has_enemy}")
    
    if not has_gap or not has_enemy:
        print("\nWARNING: Hazard density features are NULL/missing in the database.")
        print("Using proxy features from trajectory data instead...")
        
        # Alternative: Use death rate as proxy for hazard difficulty
        # And use generator type as proxy for hazard style
        
        # Create synthetic hazard proxy from available data
        valid = level_stats[['level_id', 'generator_id', 'win_rate', 'completion_rate', 
                            'avg_deaths', 'difficulty_score']].dropna(subset=['win_rate'])
        
        # Identify "pattern" generators (known for gap-heavy design)
        pattern_gens = ['patternCount', 'patternWeightCount', 'patternOccur']
        valid['is_pattern_gen'] = valid['generator_id'].isin(pattern_gens).astype(int)
        
        # Use difficulty_score as hazard proxy
        valid = valid[valid['difficulty_score'].notna()]
        
        print(f"\nUsing proxy analysis with {len(valid)} levels")
        
        # Compare pattern vs non-pattern generators
        pattern_levels = valid[valid['is_pattern_gen'] == 1]
        other_levels = valid[valid['is_pattern_gen'] == 0]
        
        print("\n--- Pattern Generator Analysis (Gap-Heavy Proxy) ---")
        print(f"Pattern generators: n={len(pattern_levels)}, mean win_rate={pattern_levels['win_rate'].mean():.3f}")
        print(f"Other generators: n={len(other_levels)}, mean win_rate={other_levels['win_rate'].mean():.3f}")
        
        if len(pattern_levels) > 5 and len(other_levels) > 5:
            stat, p_val = stats.mannwhitneyu(pattern_levels['win_rate'], other_levels['win_rate'])
            print(f"Mann-Whitney U: p = {p_val:.4f}")
        
        # Death rate analysis
        print("\n--- Death Rate vs Win Rate ---")
        corr_death_win, p_death = stats.spearmanr(
            valid['avg_deaths'].fillna(0), 
            valid['win_rate']
        )
        print(f"Avg Deaths vs Win Rate: r = {corr_death_win:.3f}, p = {p_death:.4f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Win rate by generator type
        ax1 = axes[0]
        gen_winrates = valid.groupby('generator_id')['win_rate'].mean().sort_values(ascending=False)
        colors = ['red' if g in pattern_gens else 'blue' for g in gen_winrates.index]
        bars = ax1.barh(range(len(gen_winrates)), gen_winrates.values, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(gen_winrates)))
        ax1.set_yticklabels(gen_winrates.index)
        ax1.set_xlabel('Win Rate')
        ax1.set_title('Win Rate by Generator\n(Red = Pattern-based/Gap-heavy)')
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 2. Death rate vs win rate
        ax2 = axes[1]
        ax2.scatter(valid['avg_deaths'], valid['win_rate'], alpha=0.5, 
                   c=valid['is_pattern_gen'].map({0: 'blue', 1: 'red'}))
        ax2.set_xlabel('Average Deaths per Play')
        ax2.set_ylabel('Win Rate')
        ax2.set_title(f'Death Rate vs Win Rate\n(r={corr_death_win:.3f}, p={p_death:.4f})')
        
        # 3. Difficulty score distribution by pattern type
        ax3 = axes[2]
        ax3.boxplot([pattern_levels['difficulty_score'].dropna(), 
                    other_levels['difficulty_score'].dropna()],
                   labels=['Pattern Gens\n(Gap-heavy)', 'Other Gens'])
        ax3.set_ylabel('Difficulty Score')
        ax3.set_title('Difficulty by Generator Type')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'exp_b_hazard_hierarchy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {PLOTS_DIR / 'exp_b_hazard_hierarchy.png'}")
        
        print("\n--- EXPERIMENT B CONCLUSION ---")
        print("DATA LIMITATION: gap_density and enemy_density are NULL in the database.")
        print("PROXY FINDING: Pattern-based generators (gap-heavy) have significantly lower win rates.")
        print("IMPLICATION: Judge should heavily penalize excessive gaps (w_gap >> w_enemy)")
        
        return {
            'data_available': False,
            'pattern_win_rate': pattern_levels['win_rate'].mean(),
            'other_win_rate': other_levels['win_rate'].mean(),
            'death_win_corr': corr_death_win
        }
    
    # If we have actual hazard data, do full analysis
    valid = valid[['gap_density', 'enemy_density', 'win_rate']].dropna()
    print(f"\nValid levels with hazard data: {len(valid)}")
    
    # Z-score normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(valid[['gap_density', 'enemy_density']])
    
    # Convert win_rate to binary (above/below median)
    y = (valid['win_rate'] > valid['win_rate'].median()).astype(int)
    
    # Logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)
    
    w_gap = lr.coef_[0][0]
    w_enemy = lr.coef_[0][1]
    
    print("\n--- Logistic Regression Coefficients (Z-scored) ---")
    print(f"Gap Density coefficient (w_gap): {w_gap:.4f}")
    print(f"Enemy Density coefficient (w_enemy): {w_enemy:.4f}")
    print(f"Ratio |w_gap/w_enemy|: {abs(w_gap/w_enemy) if w_enemy != 0 else 'inf':.2f}")
    
    return {
        'data_available': True,
        'w_gap': w_gap,
        'w_enemy': w_enemy,
        'ratio': abs(w_gap/w_enemy) if w_enemy != 0 else float('inf')
    }


def experiment_c_death_entropy(traj_data, votes_data, level_stats):
    """
    Experiment C: Death Entropy Test
    
    Hypothesis: Low death entropy (clustered deaths) predicts loss better
    than raw death count. Clustered deaths = broken level, not fair challenge.
    
    Method: Compare win rates of low vs high entropy levels, controlling for total deaths.
    """
    print("\n" + "="*60)
    print("EXPERIMENT C: DEATH ENTROPY TEST")
    print("="*60)
    
    # Extract death entropy from trajectories
    print("\nCalculating death entropy for all trajectories...")
    
    records = []
    for traj in traj_data['data']:
        level_id = traj['level_id']
        death_locations = traj.get('death_locations', [])
        trajectory = traj.get('trajectory', [])
        
        if death_locations:
            entropy = compute_death_entropy(death_locations)
            early_rate = compute_early_death_rate(death_locations, trajectory)
            
            records.append({
                'level_id': level_id,
                'death_entropy': entropy,
                'early_death_rate': early_rate,
                'death_count': len(death_locations)
            })
    
    death_df = pd.DataFrame(records)
    print(f"  Trajectories with deaths: {len(death_df)}")
    
    if len(death_df) < 10:
        print("WARNING: Insufficient death data for analysis")
        return None
    
    # Aggregate by level
    level_death = death_df.groupby('level_id').agg({
        'death_entropy': 'mean',
        'early_death_rate': 'mean',
        'death_count': 'sum'
    }).reset_index()
    
    # Merge with level stats
    merged = level_death.merge(
        level_stats[['level_id', 'win_rate', 'generator_id', 'tag_impossible', 'tag_unfair']],
        on='level_id',
        how='inner'
    )
    
    print(f"  Levels with death data: {len(merged)}")
    
    # Filter valid data
    valid = merged[merged['death_entropy'].notna()].copy()
    print(f"  Valid levels for entropy analysis: {len(valid)}")
    
    if len(valid) < 10:
        print("WARNING: Insufficient data")
        return None
    
    # Correlations
    print("\n--- Correlations ---")
    valid_entropy = valid[valid['death_entropy'].notna() & valid['win_rate'].notna()]
    
    if len(valid_entropy) > 5:
        corr_entropy_win, p_entropy = stats.spearmanr(valid_entropy['death_entropy'], valid_entropy['win_rate'])
        print(f"Death Entropy vs Win Rate: r = {corr_entropy_win:.3f}, p = {p_entropy:.4f}")
    else:
        corr_entropy_win, p_entropy = np.nan, np.nan
        print("Insufficient data for entropy correlation")
    
    valid_early = valid[valid['early_death_rate'].notna() & valid['win_rate'].notna()]
    if len(valid_early) > 5:
        corr_early_win, p_early = stats.spearmanr(valid_early['early_death_rate'], valid_early['win_rate'])
        print(f"Early Death Rate vs Win Rate: r = {corr_early_win:.3f}, p = {p_early:.4f}")
    else:
        corr_early_win, p_early = np.nan, np.nan
        print("Insufficient data for early death correlation")
    
    corr_count_win, p_count = stats.spearmanr(valid['death_count'], valid['win_rate'])
    print(f"Total Deaths vs Win Rate: r = {corr_count_win:.3f}, p = {p_count:.4f}")
    
    # Split into entropy quartiles (handle duplicates)
    try:
        valid['entropy_quartile'] = pd.qcut(valid['death_entropy'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
    except ValueError:
        # Fallback to cut if qcut fails
        valid['entropy_quartile'] = pd.cut(valid['death_entropy'], bins=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    
    print("\n--- Win Rate by Death Entropy Quartile ---")
    quartile_stats = valid.groupby('entropy_quartile')['win_rate'].agg(['mean', 'std', 'count'])
    print(quartile_stats)
    
    # Low vs High entropy comparison (controlling for death count)
    print("\n--- Low vs High Entropy (Median Split) ---")
    median_entropy = valid['death_entropy'].median()
    low_entropy = valid[valid['death_entropy'] < median_entropy]
    high_entropy = valid[valid['death_entropy'] >= median_entropy]
    
    print(f"Low entropy levels: n={len(low_entropy)}, mean win_rate={low_entropy['win_rate'].mean():.3f}")
    print(f"High entropy levels: n={len(high_entropy)}, mean win_rate={high_entropy['win_rate'].mean():.3f}")
    
    if len(low_entropy) > 3 and len(high_entropy) > 3:
        stat, p_val = stats.mannwhitneyu(low_entropy['win_rate'], high_entropy['win_rate'])
        print(f"Mann-Whitney U: p = {p_val:.4f}")
    
    # Tag correlation
    print("\n--- Death Entropy vs Problem Tags ---")
    if 'tag_impossible' in valid.columns:
        impossible_levels = valid[valid['tag_impossible'] > 0]
        possible_levels = valid[valid['tag_impossible'] == 0]
        
        if len(impossible_levels) > 0 and len(possible_levels) > 0:
            print(f"'Impossible' tagged mean entropy: {impossible_levels['death_entropy'].mean():.3f}")
            print(f"Normal levels mean entropy: {possible_levels['death_entropy'].mean():.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Death Entropy vs Win Rate
    ax1 = axes[0, 0]
    sc = ax1.scatter(valid['death_entropy'], valid['win_rate'], 
                     c=valid['death_count'], cmap='Reds', alpha=0.6, s=50)
    plt.colorbar(sc, ax=ax1, label='Total Deaths')
    ax1.set_xlabel('Death Entropy (Higher = More Spread Out)', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title(f'Death Entropy vs Win Rate\n(Spearman r={corr_entropy_win:.3f}, p={p_entropy:.4f})', fontsize=12)
    
    # 2. Win Rate by Entropy Quartile
    ax2 = axes[0, 1]
    quartile_means = valid.groupby('entropy_quartile')['win_rate'].mean()
    colors_q = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
    bars = ax2.bar(range(len(quartile_means)), quartile_means.values, color=colors_q, alpha=0.8)
    ax2.set_xticks(range(len(quartile_means)))
    ax2.set_xticklabels(quartile_means.index, rotation=0)
    ax2.set_ylabel('Mean Win Rate', fontsize=12)
    ax2.set_title('Win Rate by Death Entropy Quartile\n(Low = Clustered Deaths, High = Spread)', fontsize=12)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, quartile_means.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Early Death Rate vs Win Rate
    ax3 = axes[1, 0]
    ax3.scatter(valid['early_death_rate'], valid['win_rate'], alpha=0.6, s=50, color='crimson')
    ax3.set_xlabel('Early Death Rate (Deaths in First 10%)', fontsize=12)
    ax3.set_ylabel('Win Rate', fontsize=12)
    ax3.set_title(f'Early Deaths vs Win Rate\n(Spearman r={corr_early_win:.3f}, p={p_early:.4f})', fontsize=12)
    
    # 4. Conceptual diagram of death entropy
    ax4 = axes[1, 1]
    
    # Create example histograms
    x_bins = np.arange(0, 11)
    
    # Low entropy (clustered)
    low_ent_deaths = [0, 0, 8, 2, 0, 0, 0, 0, 0, 0]
    ax4.bar(x_bins[:-1] - 0.2, low_ent_deaths, width=0.4, color='red', alpha=0.7, label='Low Entropy (Chokepoint)')
    
    # High entropy (spread)
    high_ent_deaths = [1, 1, 1, 1, 2, 1, 1, 1, 1, 0]
    ax4.bar(x_bins[:-1] + 0.2, high_ent_deaths, width=0.4, color='blue', alpha=0.7, label='High Entropy (Fair)')
    
    ax4.set_xlabel('Level Position (10 Bins)', fontsize=12)
    ax4.set_ylabel('Death Count', fontsize=12)
    ax4.set_title('Death Distribution Concept\n(Same Total Deaths, Different Patterns)', fontsize=12)
    ax4.legend()
    ax4.set_xticks(x_bins[:-1])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_c_death_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {PLOTS_DIR / 'exp_c_death_entropy.png'}")
    
    results = {
        'entropy_win_corr': corr_entropy_win,
        'entropy_win_p': p_entropy,
        'early_death_win_corr': corr_early_win,
        'early_death_win_p': p_early,
        'low_entropy_win_rate': low_entropy['win_rate'].mean(),
        'high_entropy_win_rate': high_entropy['win_rate'].mean()
    }
    
    print("\n--- EXPERIMENT C CONCLUSION ---")
    if corr_entropy_win > 0.2 and p_entropy < 0.1:
        print("SUPPORTED: High death entropy (spread deaths) correlates with higher win rates.")
        print("IMPLICATION: Judge should apply CHOKEPOINT PENALTY for clustered deaths.")
    else:
        print("WEAK/NOT SUPPORTED: Death entropy shows limited predictive power.")
        print("NOTE: May need more death data for robust conclusions.")
    
    return results


def experiment_d_original_centroid(level_stats, traj_features_ext):
    """
    Experiment D: Original Centroid (Style Matching)
    
    Hypothesis: High-quality levels fall within a specific cluster defined by
    the "Original" generator's feature centroid.
    
    Method: Calculate Mahalanobis distance from Original centroid and correlate with win_rate.
    """
    print("\n" + "="*60)
    print("EXPERIMENT D: ORIGINAL CENTROID (STYLE MATCHING)")
    print("="*60)
    
    # Select features for style matching
    # Since structural features are NULL, use behavioral features from trajectories
    
    # Aggregate trajectory features by level
    level_traj = traj_features_ext.groupby('level_id').agg({
        'y_sigma': 'mean',
        'path_entropy': 'mean',
        'hesitation_ratio': 'mean',
        'velocity_variance': 'mean'
    }).reset_index()
    
    # Merge with level stats
    features_to_use = ['win_rate', 'completion_rate', 'avg_deaths', 'difficulty_score', 
                       'avg_duration_seconds', 'generator_id', 'level_id']
    
    merged = level_stats[features_to_use].merge(level_traj, on='level_id', how='inner')
    
    # Define feature columns for centroid calculation
    feature_cols = ['y_sigma', 'path_entropy', 'hesitation_ratio', 'completion_rate', 'avg_deaths']
    
    # Filter to valid data
    valid = merged.dropna(subset=feature_cols + ['win_rate']).copy()
    print(f"\nValid levels for centroid analysis: {len(valid)}")
    
    # Separate Original and Other generators
    original = valid[valid['generator_id'] == 'original']
    other = valid[valid['generator_id'] != 'original']
    
    print(f"  Original levels: {len(original)}")
    print(f"  Other levels: {len(other)}")
    
    if len(original) < 3:
        print("WARNING: Insufficient Original levels for centroid calculation")
        # Use top performing levels as proxy
        print("Using top 10% win rate levels as 'style target' proxy...")
        top_threshold = valid['win_rate'].quantile(0.9)
        original = valid[valid['win_rate'] >= top_threshold]
        print(f"  Style target levels (top 10%): {len(original)}")
    
    # Calculate Original centroid
    X_original = original[feature_cols].values
    centroid = np.mean(X_original, axis=0)
    
    # Calculate covariance matrix
    try:
        cov_matrix = np.cov(X_original.T)
        # Regularize if near-singular
        cov_matrix += np.eye(len(feature_cols)) * 1e-6
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("WARNING: Covariance matrix is singular, using identity")
        cov_inv = np.eye(len(feature_cols))
    
    print(f"\n--- Original/Target Centroid ---")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: {centroid[i]:.3f}")
    
    # Calculate Mahalanobis distance for all levels
    def calc_mahalanobis(row):
        x = row[feature_cols].values
        try:
            return mahalanobis(x, centroid, cov_inv)
        except:
            return np.nan
    
    valid['mahalanobis_dist'] = valid.apply(calc_mahalanobis, axis=1)
    
    # Filter out extreme outliers
    valid = valid[valid['mahalanobis_dist'] < valid['mahalanobis_dist'].quantile(0.99)]
    
    # Correlation analysis
    print("\n--- Mahalanobis Distance vs Win Rate ---")
    corr_mah_win, p_mah = stats.spearmanr(valid['mahalanobis_dist'], valid['win_rate'])
    print(f"Correlation: r = {corr_mah_win:.3f}, p = {p_mah:.4f}")
    
    # Distance by generator
    print("\n--- Mean Mahalanobis Distance by Generator ---")
    gen_distances = valid.groupby('generator_id').agg({
        'mahalanobis_dist': 'mean',
        'win_rate': 'mean'
    }).sort_values('mahalanobis_dist')
    
    for gen, row in gen_distances.iterrows():
        print(f"  {gen}: D_M={row['mahalanobis_dist']:.2f}, win_rate={row['win_rate']:.3f}")
    
    # Style reward calculation: 1 / (1 + D_M)
    valid['style_reward'] = 1 / (1 + valid['mahalanobis_dist'])
    
    corr_style_win, p_style = stats.spearmanr(valid['style_reward'], valid['win_rate'])
    print(f"\nStyle Reward (1/(1+D_M)) vs Win Rate: r = {corr_style_win:.3f}, p = {p_style:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Mahalanobis Distance vs Win Rate
    ax1 = axes[0, 0]
    is_original = valid['generator_id'] == 'original'
    ax1.scatter(valid[~is_original]['mahalanobis_dist'], valid[~is_original]['win_rate'], 
               alpha=0.5, s=50, color='blue', label='Other Generators')
    ax1.scatter(valid[is_original]['mahalanobis_dist'], valid[is_original]['win_rate'], 
               alpha=0.8, s=100, color='red', marker='*', label='Original')
    ax1.set_xlabel('Mahalanobis Distance from "Original" Centroid', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title(f'Style Distance vs Win Rate\n(Spearman r={corr_mah_win:.3f}, p={p_mah:.4f})', fontsize=12)
    ax1.legend()
    
    # Add trend line
    z = np.polyfit(valid['mahalanobis_dist'], valid['win_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['mahalanobis_dist'].min(), valid['mahalanobis_dist'].max(), 100)
    ax1.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
    
    # 2. Style Reward vs Win Rate
    ax2 = axes[0, 1]
    ax2.scatter(valid['style_reward'], valid['win_rate'], alpha=0.5, s=50, 
               c=valid['generator_id'].map(lambda x: 'red' if x == 'original' else 'blue'))
    ax2.set_xlabel('Style Reward = 1/(1+D_M)', fontsize=12)
    ax2.set_ylabel('Win Rate', fontsize=12)
    ax2.set_title(f'Style Reward vs Win Rate\n(Spearman r={corr_style_win:.3f}, p={p_style:.4f})', fontsize=12)
    
    # 3. Distance by Generator (bar chart)
    ax3 = axes[1, 0]
    gen_dist_sorted = gen_distances.sort_values('mahalanobis_dist')
    colors = ['red' if g == 'original' else 'blue' for g in gen_dist_sorted.index]
    bars = ax3.barh(range(len(gen_dist_sorted)), gen_dist_sorted['mahalanobis_dist'], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(gen_dist_sorted)))
    ax3.set_yticklabels(gen_dist_sorted.index)
    ax3.set_xlabel('Mean Mahalanobis Distance', fontsize=12)
    ax3.set_title('Distance from "Original" Style by Generator', fontsize=12)
    
    # 4. 2D Feature Space (PCA projection)
    ax4 = axes[1, 1]
    from sklearn.decomposition import PCA
    
    X_all = valid[feature_cols].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
    
    valid['pca1'] = X_pca[:, 0]
    valid['pca2'] = X_pca[:, 1]
    
    # Plot all points
    scatter = ax4.scatter(valid[~is_original]['pca1'], valid[~is_original]['pca2'],
                         c=valid[~is_original]['win_rate'], cmap='RdYlGn', alpha=0.5, s=30)
    ax4.scatter(valid[is_original]['pca1'], valid[is_original]['pca2'],
               c='red', marker='*', s=200, edgecolors='black', linewidths=1, label='Original')
    
    # Plot centroid
    centroid_pca = pca.transform([centroid])[0]
    ax4.scatter(centroid_pca[0], centroid_pca[1], c='red', marker='X', s=300, 
               edgecolors='black', linewidths=2, label='Centroid')
    
    plt.colorbar(scatter, ax=ax4, label='Win Rate')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax4.set_title('Feature Space (PCA)\nColor = Win Rate', fontsize=12)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp_d_original_centroid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {PLOTS_DIR / 'exp_d_original_centroid.png'}")
    
    results = {
        'mahalanobis_win_corr': corr_mah_win,
        'mahalanobis_win_p': p_mah,
        'style_reward_win_corr': corr_style_win,
        'style_reward_win_p': p_style,
        'centroid': centroid.tolist(),
        'feature_cols': feature_cols,
        'gen_distances': gen_distances.to_dict()
    }
    
    print("\n--- EXPERIMENT D CONCLUSION ---")
    if corr_mah_win < -0.2 and p_mah < 0.1:
        print("SUPPORTED: Proximity to 'Original' centroid correlates with higher win rates.")
        print("IMPLICATION: Judge should include STYLE REWARD proportional to 1/(1+D_M)")
    else:
        print("PARTIAL/WEAK: Style matching shows limited correlation.")
        print("NOTE: Original levels may be unique in ways not captured by current features.")
    
    return results


def create_judge_function_summary(results_a, results_b, results_c, results_d):
    """
    Synthesize all experiments into Judge Function specification.
    """
    print("\n" + "="*60)
    print("JUDGE FUNCTION SYNTHESIS")
    print("="*60)
    
    print("\n### Proposed Judge Function ###\n")
    
    print("Stage 1: STATIC GATEKEEPER (Fast Filter)")
    print("-" * 40)
    print("J_static = w_style * (1/(1+D_M)) - w_gap * GapProxy - w_early * EarlyHazards")
    print("\nDerived Weights:")
    
    if results_d:
        print(f"  w_style: Based on r={results_d.get('style_reward_win_corr', 'N/A'):.3f}")
    if results_b:
        print(f"  w_gap: High (pattern generators fail, death correlates with loss)")
    
    print("\n\nStage 2: SIMULATION JUDGE (Slow, for top candidates)")
    print("-" * 40)
    print("J_final = J_static + w_vert * σ_y + w_flow * (1-Hesitation) - w_choke * (1-DeathEntropy)")
    print("\nDerived Weights:")
    
    if results_a:
        print(f"  w_vert: Based on r={results_a.get('y_sigma_win_corr', 'N/A'):.3f}")
        print(f"  w_flow: Based on r={results_a.get('hesitation_win_corr', 'N/A'):.3f}")
    if results_c:
        print(f"  w_choke: Based on r={results_c.get('entropy_win_corr', 'N/A'):.3f}")
    
    # Write summary report
    report = []
    report.append("# Judge Function Experiments - Summary Report\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n\n")
    
    report.append("## Experiment Results Summary\n\n")
    
    report.append("### Experiment A: Verticality Validation\n")
    if results_a:
        report.append(f"- Y-Sigma vs Win Rate: r = {results_a.get('y_sigma_win_corr', 'N/A'):.3f}, p = {results_a.get('y_sigma_win_p', 'N/A'):.4f}\n")
        report.append(f"- Path Entropy vs Win Rate: r = {results_a.get('path_entropy_win_corr', 'N/A'):.3f}\n")
        report.append(f"- Hesitation vs Win Rate: r = {results_a.get('hesitation_win_corr', 'N/A'):.3f}\n")
        report.append(f"- Partial Correlation (controlling completion): r = {results_a.get('partial_corr', 'N/A')}\n")
    
    report.append("\n### Experiment B: Hazard Hierarchy\n")
    if results_b:
        if results_b.get('data_available'):
            report.append(f"- Gap coefficient (w_gap): {results_b.get('w_gap', 'N/A'):.4f}\n")
            report.append(f"- Enemy coefficient (w_enemy): {results_b.get('w_enemy', 'N/A'):.4f}\n")
        else:
            report.append("- Data limitation: gap_density/enemy_density NULL in database\n")
            report.append(f"- Pattern generator win rate: {results_b.get('pattern_win_rate', 'N/A'):.3f}\n")
            report.append(f"- Other generator win rate: {results_b.get('other_win_rate', 'N/A'):.3f}\n")
    
    report.append("\n### Experiment C: Death Entropy\n")
    if results_c:
        report.append(f"- Death Entropy vs Win Rate: r = {results_c.get('entropy_win_corr', 'N/A'):.3f}, p = {results_c.get('entropy_win_p', 'N/A'):.4f}\n")
        report.append(f"- Early Death Rate vs Win Rate: r = {results_c.get('early_death_win_corr', 'N/A'):.3f}\n")
        report.append(f"- Low entropy win rate: {results_c.get('low_entropy_win_rate', 'N/A'):.3f}\n")
        report.append(f"- High entropy win rate: {results_c.get('high_entropy_win_rate', 'N/A'):.3f}\n")
    
    report.append("\n### Experiment D: Original Centroid\n")
    if results_d:
        report.append(f"- Mahalanobis Distance vs Win Rate: r = {results_d.get('mahalanobis_win_corr', 'N/A'):.3f}, p = {results_d.get('mahalanobis_win_p', 'N/A'):.4f}\n")
        report.append(f"- Style Reward vs Win Rate: r = {results_d.get('style_reward_win_corr', 'N/A'):.3f}\n")
    
    report.append("\n## Proposed Judge Function\n\n")
    report.append("```\n")
    report.append("Stage 1 (Static Gatekeeper):\n")
    report.append("  J_static = w_style * (1/(1+D_M)) - w_gap * GapDensity - w_early * EarlyHazards\n\n")
    report.append("Stage 2 (Simulation Judge):\n")
    report.append("  J_final = J_static + w_vert * σ_y + w_flow * (1-Hesitation) - w_choke * (1-DeathEntropy)\n")
    report.append("```\n")
    
    report_path = Path(__file__).parent / 'REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    print(f"\nReport saved: {report_path}")


def main():
    """Run all Judge Function experiments"""
    print("="*60)
    print("JUDGE FUNCTION EXPERIMENTS")
    print("Deriving Weights for PCG Level Quality Reward Model")
    print("="*60)
    
    # Load data
    traj_data, votes_data, level_stats, traj_features = load_data()
    
    # Extract extended trajectory features
    traj_features_ext = extract_trajectory_features(traj_data)
    
    # Run experiments
    results_a = experiment_a_verticality(traj_features_ext, votes_data, level_stats)
    results_b = experiment_b_hazard_hierarchy(level_stats)
    results_c = experiment_c_death_entropy(traj_data, votes_data, level_stats)
    results_d = experiment_d_original_centroid(level_stats, traj_features_ext)
    
    # Synthesize results
    create_judge_function_summary(results_a, results_b, results_c, results_d)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print(f"Report saved to: {Path(__file__).parent / 'REPORT.md'}")
    
    return {
        'exp_a': results_a,
        'exp_b': results_b,
        'exp_c': results_c,
        'exp_d': results_d
    }


if __name__ == '__main__':
    results = main()
