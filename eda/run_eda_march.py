"""
PCG Arena EDA - March 2026 Refresh
====================================
Comprehensive EDA on the updated PCG Arena dataset (~2x data).
Generates all plots and a summary report.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import Counter
import shutil
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "eda" / "data_23_03"
PLOTS_DIR = BASE_DIR / "eda" / "plots_23_03"
LATEX_IMG_DIR = BASE_DIR / "latex" / "img"
REPORT_PATH = BASE_DIR / "eda" / "report_23_03.md"

PLOTS_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all data files."""
    data = {}
    for name in ['level-stats', 'votes', 'player-profiles', 'trajectories']:
        fpath = DATA_DIR / f"pcg-arena-{name}-2026-03-23.json"
        with open(fpath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        data[name] = raw['data']
    
    level_stats = pd.DataFrame(data['level-stats'])
    votes = pd.DataFrame(data['votes'])
    players = pd.DataFrame(data['player-profiles'])
    trajectories = pd.DataFrame(data['trajectories'])
    
    print(f"Levels:       {len(level_stats)}")
    print(f"Votes:        {len(votes)} (total in API: may be more)")
    print(f"Players:      {len(players)}")
    print(f"Trajectories: {len(trajectories)}")
    
    return level_stats, votes, players, trajectories


def extract_telemetry(votes_df):
    """Flatten telemetry from votes into per-play records."""
    records = []
    for _, row in votes_df.iterrows():
        telemetry = row.get('telemetry', {})
        if not telemetry or not isinstance(telemetry, dict):
            continue
        for side in ['left', 'right']:
            sd = telemetry.get(side, {})
            if not sd or not isinstance(sd, dict):
                continue
            death_locs = sd.get('death_locations', [])
            if death_locs is None:
                death_locs = []
            death_causes = [d.get('cause', 'unknown') for d in death_locs if isinstance(d, dict)]
            cc = Counter(death_causes)
            
            won = (row['result'] == 'LEFT' and side == 'left') or \
                  (row['result'] == 'RIGHT' and side == 'right')
            lost = (row['result'] == 'LEFT' and side == 'right') or \
                   (row['result'] == 'RIGHT' and side == 'left')
            
            tags = row.get(f'{side}_tags', [])
            if tags is None:
                tags = []
            
            records.append({
                'vote_id': row['vote_id'],
                'player_id': row.get('player_id'),
                'side': side,
                'level_id': row.get(f'{side}_level_id'),
                'generator_id': row.get(f'{side}_generator_id'),
                'result': row['result'],
                'won': won,
                'lost': lost,
                'tied': row['result'] == 'TIE',
                'duration_seconds': sd.get('duration_seconds', 0),
                'completed': sd.get('completed', False),
                'deaths': sd.get('deaths', 0),
                'coins_collected': sd.get('coins_collected', 0),
                'enemies_stomped': sd.get('enemies_stomped', 0),
                'jumps': sd.get('jumps', 0),
                'death_by_enemy': cc.get('enemy', 0),
                'death_by_fall': cc.get('fall', 0),
                'tags': tags,
                'tag_fun': 'fun' in tags,
                'tag_boring': 'boring' in tags,
                'tag_too_hard': 'too_hard' in tags,
                'tag_too_easy': 'too_easy' in tags,
                'tag_creative': 'creative' in tags,
                'tag_good_flow': 'good_flow' in tags,
                'tag_unfair': 'unfair' in tags,
                'tag_confusing': 'confusing' in tags,
            })
    return pd.DataFrame(records)


def compute_generator_stats(level_stats, telemetry):
    """Aggregate stats per generator from level_stats."""
    tag_cols = [c for c in level_stats.columns if c.startswith('tag_')]
    
    agg_dict = {
        'level_id': 'count',
        'times_shown': 'sum',
        'times_won': 'sum',
        'times_lost': 'sum',
        'times_tied': 'sum',
        'win_rate': 'mean',
        'completion_rate': 'mean',
        'avg_deaths': 'mean',
        'avg_duration_seconds': 'mean',
    }
    for tc in tag_cols:
        agg_dict[tc] = 'sum'
    
    gs = level_stats.groupby('generator_id').agg(agg_dict).rename(
        columns={'level_id': 'num_levels'}
    )
    gs['total_games'] = gs['times_won'] + gs['times_lost'] + gs['times_tied']
    gs['overall_win_rate'] = gs['times_won'] / gs['total_games'].replace(0, np.nan)
    
    # From telemetry
    if len(telemetry) > 0:
        tg = telemetry.groupby('generator_id').agg({
            'vote_id': 'count',
            'won': 'mean',
            'duration_seconds': 'mean',
            'deaths': 'mean',
            'completed': 'mean',
        }).rename(columns={
            'vote_id': 'num_plays',
            'won': 'telemetry_win_rate',
            'duration_seconds': 'tel_avg_duration',
            'deaths': 'tel_avg_deaths',
            'completed': 'tel_completion_rate',
        })
        gs = gs.join(tg, how='left')
    
    return gs.reset_index()


# =============================================================================
# PLOTS
# =============================================================================

def plot_generator_rankings(gen_stats):
    """Bar chart of generator win rates (overall_win_rate from level_stats)."""
    df = gen_stats.sort_values('overall_win_rate', ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2ecc71' if wr > 0.6 else '#e74c3c' if wr < 0.4 else '#f39c12' 
              for wr in df['overall_win_rate']]
    bars = ax.barh(df['generator_id'], df['overall_win_rate'], color=colors, edgecolor='white')
    
    for bar, val, games in zip(bars, df['overall_win_rate'], df['total_games']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%} ({int(games)} games)', va='center', fontsize=9)
    
    ax.set_xlabel('Win Rate')
    ax.set_title('Generator Rankings by Win Rate', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generator_rankings.png", dpi=150, bbox_inches='tight')
    plt.close()
    return df


def plot_h1_difficulty_vs_winrate(level_stats):
    """H1: Win rate vs difficulty (avg deaths)."""
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    df['death_rate'] = df['avg_deaths']
    df = df[df['death_rate'] <= 10]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter
    ax = axes[0]
    ax.scatter(df['death_rate'], df['win_rate'], alpha=0.4, s=20)
    valid = df[['death_rate', 'win_rate']].dropna()
    if len(valid) > 10:
        z = np.polyfit(valid['death_rate'], valid['win_rate'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(valid['death_rate'].min(), valid['death_rate'].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', lw=2, label='Quadratic fit')
    corr, pval = stats.spearmanr(valid['death_rate'], valid['win_rate'])
    ax.set_xlabel('Average Deaths per Play')
    ax.set_ylabel('Win Rate')
    ax.set_title(f'H1: Win Rate vs Difficulty (r={corr:.3f}, p={pval:.4f})')
    ax.legend()
    
    # Boxplot by bins
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]
    labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
    df['diff_bin'] = pd.cut(df['death_rate'], bins=bins, labels=labels)
    ax2 = axes[1]
    df.dropna(subset=['diff_bin']).boxplot(column='win_rate', by='diff_bin', ax=ax2)
    ax2.set_xlabel('Deaths per Play (binned)')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate by Difficulty Bin')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h1_difficulty_vs_winrate.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'corr': corr, 'p': pval, 'n': len(valid)}


def plot_h2_tags_vs_winrate(level_stats):
    """H2: Tag rates vs win rate correlation."""
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    tag_cols = ['tag_fun', 'tag_creative', 'tag_boring', 'tag_too_hard', 
                'tag_too_easy', 'tag_good_flow', 'tag_unfair', 'tag_confusing']
    existing_tags = [c for c in tag_cols if c in df.columns]
    
    # Compute tag rates
    for tc in existing_tags:
        df[f'{tc}_rate'] = df[tc] / df['times_shown'].replace(0, 1)
    
    rate_cols = [f'{tc}_rate' for tc in existing_tags]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations = {}
    for rc in rate_cols:
        valid = df[[rc, 'win_rate']].dropna()
        if len(valid) > 10:
            corr, pval = stats.spearmanr(valid[rc], valid['win_rate'])
            correlations[rc.replace('tag_', '').replace('_rate', '')] = (corr, pval)
    
    names = list(correlations.keys())
    corrs = [correlations[n][0] for n in names]
    pvals = [correlations[n][1] for n in names]
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in pvals]
    
    bars = ax.barh(names, corrs, color=colors, edgecolor='white')
    for bar, c, p in zip(bars, corrs, pvals):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(bar.get_width() + 0.01 if c >= 0 else bar.get_width() - 0.06,
                bar.get_y() + bar.get_height()/2,
                f'r={c:.3f}{sig}', va='center', fontsize=9)
    
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Spearman Correlation with Win Rate')
    ax.set_title('H2: Tag Rate Correlations with Win Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h2_tags_vs_winrate.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return correlations


def plot_h3_telemetry_winners_vs_losers(telemetry):
    """H3: Compare telemetry between won and lost plays."""
    won = telemetry[telemetry['won'] == True].copy()
    lost = telemetry[telemetry['lost'] == True].copy()
    
    metrics = ['duration_seconds', 'deaths', 'completed', 'coins_collected', 'jumps']
    results = {}
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for i, m in enumerate(metrics):
        ax = axes[i]
        w_vals = won[m].dropna().astype(float)
        l_vals = lost[m].dropna().astype(float)
        
        ax.boxplot([w_vals, l_vals], labels=['Won', 'Lost'])
        stat, pval = stats.mannwhitneyu(w_vals, l_vals, alternative='two-sided') if len(w_vals) > 5 and len(l_vals) > 5 else (np.nan, np.nan)
        ax.set_title(f'{m}\np={pval:.4f}' if not np.isnan(pval) else m, fontsize=10)
        results[m] = {'won_mean': w_vals.mean(), 'lost_mean': l_vals.mean(), 'p': pval}
    
    plt.suptitle('H3: Telemetry — Won vs Lost Plays', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h3_telemetry_won_vs_lost.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def plot_tag_distribution(telemetry):
    """Distribution of tags across all plays."""
    tag_cols = [c for c in telemetry.columns if c.startswith('tag_') and telemetry[c].dtype == bool]
    
    counts = {c.replace('tag_', ''): telemetry[c].sum() for c in tag_cols}
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.keys(), counts.values(), color=sns.color_palette("husl", len(counts)))
    ax.set_ylabel('Count')
    ax.set_title('Tag Distribution Across All Plays', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tag_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return counts


def plot_tag_by_generator(telemetry):
    """Tag usage heatmap by generator."""
    tag_cols = [c for c in telemetry.columns if c.startswith('tag_') and telemetry[c].dtype == bool]
    
    if not tag_cols:
        return
    
    tag_rates = telemetry.groupby('generator_id')[tag_cols].mean()
    tag_rates.columns = [c.replace('tag_', '') for c in tag_rates.columns]
    
    # Sort by overall win rate
    gen_order = telemetry.groupby('generator_id')['won'].mean().sort_values(ascending=False).index
    tag_rates = tag_rates.reindex(gen_order)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(tag_rates, annot=True, fmt='.2f', cmap='RdYlGn', center=0.1,
                ax=ax, linewidths=0.5)
    ax.set_title('Tag Rates by Generator', fontsize=14, fontweight='bold')
    ax.set_ylabel('Generator')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tag_by_generator.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_h6_tag_validation(telemetry):
    """H6: Validate tags against telemetry metrics."""
    tag_metric_pairs = [
        ('tag_fun', 'completed', 'Fun → Completion'),
        ('tag_too_hard', 'deaths', 'Too Hard → Deaths'),
        ('tag_too_easy', 'completed', 'Too Easy → Completion'),
        ('tag_creative', 'won', 'Creative → Wins'),
        ('tag_boring', 'duration_seconds', 'Boring → Duration'),
        ('tag_good_flow', 'completed', 'Good Flow → Completion'),
    ]
    
    results = {}
    valid_pairs = [(t, m, l) for t, m, l in tag_metric_pairs if t in telemetry.columns]
    
    if not valid_pairs:
        return results
    
    n_pairs = len(valid_pairs)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (tag, metric, label) in enumerate(valid_pairs):
        ax = axes[i]
        tagged = telemetry[telemetry[tag] == True][metric].dropna().astype(float)
        not_tagged = telemetry[telemetry[tag] == False][metric].dropna().astype(float)
        
        if len(tagged) > 2 and len(not_tagged) > 2:
            ax.boxplot([not_tagged, tagged], labels=['No Tag', 'Tagged'])
            stat, pval = stats.mannwhitneyu(tagged, not_tagged, alternative='two-sided')
            ax.set_title(f'{label}\np={pval:.4f}', fontsize=11)
            results[label] = {
                'tagged_mean': tagged.mean(),
                'not_tagged_mean': not_tagged.mean(),
                'tagged_n': len(tagged),
                'not_tagged_n': len(not_tagged),
                'p': pval,
            }
        else:
            ax.set_title(f'{label}\n(insufficient data)')
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('H6: Tag Validation Against Telemetry', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h6_tag_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def plot_player_engagement(players):
    """Player engagement distribution."""
    df = players[players['total_votes'] > 0].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Votes per player
    ax = axes[0]
    ax.hist(df['total_votes'], bins=30, color='steelblue', edgecolor='white')
    ax.set_xlabel('Total Votes')
    ax.set_ylabel('Count')
    ax.set_title(f'Votes per Player (n={len(df)})')
    ax.axvline(df['total_votes'].median(), color='red', linestyle='--', label=f'median={df["total_votes"].median():.0f}')
    ax.legend()
    
    # Sessions per player
    ax = axes[1]
    ax.hist(df['total_sessions'].dropna(), bins=20, color='coral', edgecolor='white')
    ax.set_xlabel('Total Sessions')
    ax.set_title('Sessions per Player')
    
    # Cumulative vote contribution
    ax = axes[2]
    sorted_votes = df['total_votes'].sort_values(ascending=False).values
    cum_pct = np.cumsum(sorted_votes) / sorted_votes.sum() * 100
    ax.plot(range(1, len(cum_pct)+1), cum_pct, 'b-', lw=2)
    ax.set_xlabel('Number of Players (ranked by activity)')
    ax.set_ylabel('Cumulative % of Votes')
    ax.set_title('Vote Concentration')
    ax.axhline(80, color='red', linestyle='--', alpha=0.5, label='80%')
    # Find how many players contribute 80%
    n_80 = np.searchsorted(cum_pct, 80) + 1
    ax.axvline(n_80, color='red', linestyle='--', alpha=0.5)
    ax.legend(title=f'{n_80} players = 80% votes')
    
    plt.suptitle('Player Engagement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "player_engagement.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'total_players': len(players),
        'active_players': len(df),
        'median_votes': df['total_votes'].median(),
        'mean_votes': df['total_votes'].mean(),
        'top_player_votes': df['total_votes'].max(),
        'n_80pct': n_80,
    }


def plot_generator_comparison_radar(gen_stats):
    """Radar chart comparing top generators."""
    top_gens = gen_stats.nlargest(6, 'overall_win_rate')
    
    metrics = ['overall_win_rate', 'completion_rate', 'avg_duration_seconds']
    # Add tag-based metrics if available
    for tc in ['tag_fun', 'tag_creative', 'tag_good_flow']:
        if tc in top_gens.columns:
            top_gens[f'{tc}_rate_calc'] = top_gens[tc] / top_gens['times_shown'].replace(0, 1)
            metrics.append(f'{tc}_rate_calc')
    
    # Normalize each metric to 0-1
    norm_data = top_gens[metrics].copy()
    for col in metrics:
        mn, mx = norm_data[col].min(), norm_data[col].max()
        if mx > mn:
            norm_data[col] = (norm_data[col] - mn) / (mx - mn)
        else:
            norm_data[col] = 0.5
    
    labels = [m.replace('overall_', '').replace('_rate_calc', ' rate').replace('_', ' ').title() for m in metrics]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for idx, (_, row) in enumerate(norm_data.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        gen_name = top_gens.iloc[idx]['generator_id']
        ax.plot(angles, values, 'o-', lw=2, label=gen_name, markersize=4)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Top 6 Generators Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "generator_radar.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(level_stats):
    """Correlation matrix of numeric level features."""
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    numeric_cols = ['win_rate', 'completion_rate', 'avg_deaths', 'avg_duration_seconds',
                    'difficulty_score', 'times_shown']
    tag_cols = [c for c in df.columns if c.startswith('tag_')]
    
    # Compute tag rates
    for tc in tag_cols:
        df[f'{tc}_rate'] = df[tc] / df['times_shown'].replace(0, 1)
    
    all_cols = numeric_cols + [f'{tc}_rate' for tc in tag_cols if f'{tc}_rate' in df.columns]
    existing = [c for c in all_cols if c in df.columns]
    
    corr_matrix = df[existing].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Level Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_completion_vs_winrate(level_stats):
    """Scatter: completion rate vs win rate by generator."""
    df = level_stats[level_stats['times_shown'] >= 3].copy()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    generators = df['generator_id'].unique()
    colors = sns.color_palette("husl", len(generators))
    
    for gen, color in zip(generators, colors):
        gdf = df[df['generator_id'] == gen]
        ax.scatter(gdf['completion_rate'], gdf['win_rate'], alpha=0.5, s=25,
                   color=color, label=gen)
    
    corr, pval = stats.spearmanr(df['completion_rate'].dropna(), df['win_rate'].dropna())
    ax.set_xlabel('Completion Rate')
    ax.set_ylabel('Win Rate')
    ax.set_title(f'Completion Rate vs Win Rate (r={corr:.3f}, p={pval:.4f})', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "completion_vs_winrate.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'corr': corr, 'p': pval}


def plot_global_distributions(level_stats, gen_stats):
    """Global distribution plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Win rate distribution
    ax = axes[0, 0]
    df = level_stats[level_stats['times_shown'] >= 3]
    ax.hist(df['win_rate'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Win Rate')
    ax.set_ylabel('Count')
    ax.set_title(f'Level Win Rate Distribution (n={len(df)})')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # Deaths distribution
    ax = axes[0, 1]
    ax.hist(df['avg_deaths'].clip(upper=5), bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Avg Deaths per Play')
    ax.set_title('Death Rate Distribution')
    
    # Games per generator
    ax = axes[1, 0]
    gs = gen_stats.sort_values('total_games', ascending=True)
    ax.barh(gs['generator_id'], gs['total_games'], color='teal', edgecolor='white')
    ax.set_xlabel('Total Games')
    ax.set_title('Games per Generator')
    
    # Completion rate by generator
    ax = axes[1, 1]
    gs2 = gen_stats.sort_values('completion_rate', ascending=True)
    ax.barh(gs2['generator_id'], gs2['completion_rate'], color='goldenrod', edgecolor='white')
    ax.set_xlabel('Avg Completion Rate')
    ax.set_title('Completion Rate by Generator')
    
    plt.suptitle('Global Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "global_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_duration_analysis(telemetry):
    """Play duration analysis by generator and outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Duration by generator
    ax = axes[0]
    gen_order = telemetry.groupby('generator_id')['duration_seconds'].median().sort_values().index
    telemetry_filtered = telemetry[telemetry['duration_seconds'] < 120]
    data_by_gen = [telemetry_filtered[telemetry_filtered['generator_id'] == g]['duration_seconds'].values 
                   for g in gen_order]
    bp = ax.boxplot(data_by_gen, labels=gen_order, vert=True)
    ax.set_xticklabels(gen_order, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Play Duration by Generator')
    
    # Duration won vs lost
    ax = axes[1]
    won_dur = telemetry_filtered[telemetry_filtered['won'] == True]['duration_seconds']
    lost_dur = telemetry_filtered[telemetry_filtered['lost'] == True]['duration_seconds']
    ax.boxplot([won_dur, lost_dur], labels=['Won', 'Lost'])
    stat, pval = stats.mannwhitneyu(won_dur, lost_dur, alternative='two-sided') if len(won_dur) > 5 and len(lost_dur) > 5 else (np.nan, np.nan)
    ax.set_ylabel('Duration (seconds)')
    ax.set_title(f'Duration: Won vs Lost (p={pval:.4f})' if not np.isnan(pval) else 'Duration: Won vs Lost')
    
    plt.suptitle('Play Duration Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "duration_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PCG Arena EDA - March 23, 2026 Data Refresh")
    print("=" * 70)
    
    # Load
    print("\n[1/8] Loading data...")
    level_stats, votes, players, trajectories = load_data()
    
    print("\n[2/8] Extracting telemetry...")
    telemetry = extract_telemetry(votes)
    print(f"  Telemetry records: {len(telemetry)}")
    
    print("\n[3/8] Computing generator stats...")
    gen_stats = compute_generator_stats(level_stats, telemetry)
    
    # Generate all plots
    print("\n[4/8] Generating plots...")
    
    print("  - Generator rankings...")
    ranking_df = plot_generator_rankings(gen_stats)
    
    print("  - Global distributions...")
    plot_global_distributions(level_stats, gen_stats)
    
    print("  - H1: Difficulty vs win rate...")
    h1_results = plot_h1_difficulty_vs_winrate(level_stats)
    
    print("  - H2: Tags vs win rate...")
    h2_results = plot_h2_tags_vs_winrate(level_stats)
    
    print("  - H3: Telemetry won vs lost...")
    h3_results = plot_h3_telemetry_winners_vs_losers(telemetry)
    
    print("  - Tag distribution...")
    tag_counts = plot_tag_distribution(telemetry)
    
    print("  - Tag by generator heatmap...")
    plot_tag_by_generator(telemetry)
    
    print("  - H6: Tag validation...")
    h6_results = plot_h6_tag_validation(telemetry)
    
    print("  - Player engagement...")
    player_stats = plot_player_engagement(players)
    
    print("  - Generator radar chart...")
    plot_generator_comparison_radar(gen_stats)
    
    print("  - Correlation matrix...")
    plot_correlation_matrix(level_stats)
    
    print("  - Completion vs win rate...")
    comp_results = plot_completion_vs_winrate(level_stats)
    
    print("  - Duration analysis...")
    plot_duration_analysis(telemetry)
    
    # Compute key statistics for report
    print("\n[5/8] Computing summary statistics...")
    
    # Generator ranking table
    gen_ranking = gen_stats.sort_values('overall_win_rate', ascending=False)[
        ['generator_id', 'overall_win_rate', 'total_games', 'num_levels',
         'completion_rate', 'avg_deaths', 'avg_duration_seconds']
    ].copy()
    gen_ranking['overall_win_rate'] = gen_ranking['overall_win_rate'].map(lambda x: f'{x:.1%}')
    gen_ranking['completion_rate'] = gen_ranking['completion_rate'].map(lambda x: f'{x:.1%}')
    gen_ranking['avg_deaths'] = gen_ranking['avg_deaths'].map(lambda x: f'{x:.2f}')
    gen_ranking['avg_duration_seconds'] = gen_ranking['avg_duration_seconds'].map(lambda x: f'{x:.1f}')
    
    total_votes_api = 1109  # from API total field
    
    # Build report
    print("\n[6/8] Generating report...")
    
    report = f"""# PCG Arena EDA Report — March 23, 2026

## Dataset Summary

| Metric | Previous (Jan 28) | Current (Mar 23) | Growth |
|--------|-------------------|-------------------|--------|
| Levels | 748 | {len(level_stats)} | {len(level_stats)/748:.1f}x |
| Total Votes | 571 | {total_votes_api} | {total_votes_api/571:.1f}x |
| Players | 27 | {len(players)} | {len(players)/27:.1f}x |
| Trajectories | 100 | {trajectories.shape[0]} (fetched) | — |
| Telemetry records | ~1,142 | {len(telemetry)} | {len(telemetry)/1142:.1f}x |

## Generator Rankings

| Rank | Generator | Win Rate | Games | Levels | Completion | Avg Deaths | Avg Duration |
|------|-----------|----------|-------|--------|------------|------------|-------------|
"""
    for i, (_, row) in enumerate(gen_ranking.iterrows(), 1):
        report += f"| {i} | {row['generator_id']} | {row['overall_win_rate']} | {int(row['total_games'])} | {int(row['num_levels'])} | {row['completion_rate']} | {row['avg_deaths']} | {row['avg_duration_seconds']}s |\n"
    
    report += f"""
## RQ1: What Makes a Good Level?

### H1: Difficulty vs Win Rate
- **Result**: Spearman r = {h1_results['corr']:.3f}, p = {h1_results['p']:.4f} (n = {h1_results['n']})
- **Interpretation**: {'Significant negative correlation — easier levels win more.' if h1_results['p'] < 0.05 else 'No significant relationship.'}
- **Flow channel hypothesis**: {'Not supported. Monotonic negative relationship, not inverted-U.' if h1_results['corr'] < 0 and h1_results['p'] < 0.05 else 'Inconclusive.'}

### H2: Tag Correlations with Win Rate
| Tag | Spearman r | p-value | Significant |
|-----|-----------|---------|-------------|
"""
    for tag, (corr, pval) in sorted(h2_results.items(), key=lambda x: abs(x[1][0]), reverse=True):
        sig = '✅' if pval < 0.05 else '❌'
        report += f"| {tag} | {corr:.3f} | {pval:.4f} | {sig} |\n"
    
    report += f"""
### H3: Telemetry — Won vs Lost Plays
| Metric | Won Mean | Lost Mean | p-value | Significant |
|--------|----------|-----------|---------|-------------|
"""
    for metric, res in h3_results.items():
        sig = '✅' if res['p'] < 0.05 else '❌'
        report += f"| {metric} | {res['won_mean']:.3f} | {res['lost_mean']:.3f} | {res['p']:.4f} | {sig} |\n"
    
    report += f"""
### Completion Rate vs Win Rate
- Spearman r = {comp_results['corr']:.3f}, p = {comp_results['p']:.4f}

## RQ3: Tag Validation (H6)

| Tag → Metric | Tagged Mean | Not Tagged Mean | Tagged n | p-value | Significant |
|-------------|-------------|-----------------|----------|---------|-------------|
"""
    for label, res in h6_results.items():
        sig = '✅' if res['p'] < 0.05 else '❌'
        report += f"| {label} | {res['tagged_mean']:.3f} | {res['not_tagged_mean']:.3f} | {res['tagged_n']} | {res['p']:.4f} | {sig} |\n"
    
    report += f"""
## Player Engagement

| Metric | Value |
|--------|-------|
| Total registered players | {player_stats['total_players']} |
| Active players (≥1 vote) | {player_stats['active_players']} |
| Median votes per player | {player_stats['median_votes']:.0f} |
| Mean votes per player | {player_stats['mean_votes']:.1f} |
| Most active player | {player_stats['top_player_votes']:.0f} votes |
| Players contributing 80% of votes | {player_stats['n_80pct']} |

## Tag Distribution

| Tag | Count |
|-----|-------|
"""
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"| {tag} | {count} |\n"
    
    report += f"""
## Plots Generated

All plots saved to `eda/plots_23_03/`:
- `generator_rankings.png` — Bar chart of win rates
- `global_distributions.png` — Win rate, deaths, games, completion distributions
- `h1_difficulty_vs_winrate.png` — Difficulty vs win rate scatter + boxplot
- `h2_tags_vs_winrate.png` — Tag rate correlations with win rate
- `h3_telemetry_won_vs_lost.png` — Telemetry comparison: won vs lost
- `tag_distribution.png` — Tag frequency distribution
- `tag_by_generator.png` — Tag rate heatmap by generator
- `h6_tag_validation.png` — Tag validation against objective telemetry
- `player_engagement.png` — Player activity distributions
- `generator_radar.png` — Top 6 generators radar comparison
- `correlation_matrix.png` — Level feature correlation matrix
- `completion_vs_winrate.png` — Completion rate vs win rate scatter
- `duration_analysis.png` — Play duration by generator and outcome

---
*Generated: March 23, 2026*
"""
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report written to: {REPORT_PATH}")
    
    # Copy plots to latex/img with eda_ prefix
    print("\n[7/8] Copying plots to latex/img/...")
    for plot_file in PLOTS_DIR.glob("*.png"):
        dest = LATEX_IMG_DIR / f"eda_{plot_file.name}"
        shutil.copy2(plot_file, dest)
        print(f"  Copied: {plot_file.name} → eda_{plot_file.name}")
    
    print("\n[8/8] Done!")
    print(f"\nTotal plots: {len(list(PLOTS_DIR.glob('*.png')))}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
