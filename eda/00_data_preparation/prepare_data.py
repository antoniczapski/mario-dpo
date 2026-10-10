"""
PCG Arena EDA - Data Preparation Script
========================================
This script loads all raw JSON exports and creates clean DataFrames
with extracted features for downstream analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

# Paths
EDA_DIR = Path(__file__).parent.parent
DATA_DIR = EDA_DIR / "new_data"  # Use new_data directory for fresh data
OUTPUT_DIR = EDA_DIR / "00_data_preparation"

def load_json(filename_pattern: str) -> dict:
    """Load a JSON file matching the pattern."""
    files = list(DATA_DIR.glob(filename_pattern))
    if not files:
        # Fallback to main eda directory
        files = list(EDA_DIR.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {filename_pattern}")
    with open(files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def load_level_stats() -> pd.DataFrame:
    """Load level statistics."""
    data = load_json("pcg-arena-level-stats-*.json")
    df = pd.DataFrame(data['data'])
    print(f"Loaded {len(df)} level stats records")
    return df

def load_votes() -> pd.DataFrame:
    """Load vote records with telemetry."""
    data = load_json("pcg-arena-votes-*.json")
    df = pd.DataFrame(data['data'])
    print(f"Loaded {len(df)} vote records")
    return df

def load_trajectories() -> pd.DataFrame:
    """Load trajectory data."""
    data = load_json("pcg-arena-trajectories-*.json")
    df = pd.DataFrame(data['data'])
    print(f"Loaded {len(df)} trajectory records")
    return df

def load_player_profiles() -> pd.DataFrame:
    """Load player profiles."""
    data = load_json("pcg-arena-player-profiles-*.json")
    df = pd.DataFrame(data['data'])
    print(f"Loaded {len(df)} player profiles")
    return df

def extract_telemetry_features(votes_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from telemetry nested in votes."""
    records = []
    
    for _, row in votes_df.iterrows():
        telemetry = row.get('telemetry', {})
        if not telemetry:
            continue
            
        for side in ['left', 'right']:
            side_data = telemetry.get(side, {})
            if not side_data:
                continue
                
            level_id = row.get(f'{side}_level_id')
            generator_id = row.get(f'{side}_generator_id')
            
            # Extract death locations
            death_locs = side_data.get('death_locations', [])
            death_x_positions = [d.get('x', 0) for d in death_locs if d]
            death_causes = [d.get('cause', 'unknown') for d in death_locs if d]
            
            # Count death causes
            cause_counts = Counter(death_causes)
            
            record = {
                'vote_id': row['vote_id'],
                'player_id': row.get('player_id'),
                'session_id': row.get('session_id'),
                'side': side,
                'level_id': level_id,
                'generator_id': generator_id,
                'result': row['result'],
                'won': (row['result'] == 'LEFT' and side == 'left') or (row['result'] == 'RIGHT' and side == 'right'),
                'lost': (row['result'] == 'LEFT' and side == 'right') or (row['result'] == 'RIGHT' and side == 'left'),
                'tied': row['result'] == 'TIE',
                'duration_seconds': side_data.get('duration_seconds', 0),
                'completed': side_data.get('completed', False),
                'deaths': side_data.get('deaths', 0),
                'coins_collected': side_data.get('coins_collected', 0),
                'enemies_stomped': side_data.get('enemies_stomped', 0),
                'enemies_fire_killed': side_data.get('enemies_fire_killed', 0),
                'enemies_shell_killed': side_data.get('enemies_shell_killed', 0),
                'jumps': side_data.get('jumps', 0),
                'powerups_mushroom': side_data.get('powerups_mushroom', 0),
                'powerups_flower': side_data.get('powerups_flower', 0),
                'death_by_enemy': cause_counts.get('enemy', 0),
                'death_by_fall': cause_counts.get('fall', 0),
                'death_by_timeout': cause_counts.get('timeout', 0),
                'avg_death_x': np.mean(death_x_positions) if death_x_positions else np.nan,
                'max_death_x': max(death_x_positions) if death_x_positions else np.nan,
                'tags': row.get(f'{side}_tags', []),
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} telemetry records from votes")
    return df

def extract_trajectory_features(traj_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from raw trajectory data."""
    records = []
    
    for _, row in traj_df.iterrows():
        traj = row.get('trajectory', [])
        if not traj or len(traj) < 2:
            continue
        
        # Extract x, y positions
        x_positions = [p.get('x', 0) for p in traj]
        y_positions = [p.get('y', 0) for p in traj]
        states = [p.get('state', 0) for p in traj]
        
        # Calculate trajectory features
        x_arr = np.array(x_positions)
        y_arr = np.array(y_positions)
        
        # Progress (max x reached)
        max_x = np.max(x_arr)
        
        # Spatial coverage (unique positions visited)
        unique_positions = len(set(zip(
            (x_arr // 16).astype(int),  # Tile-level granularity
            (y_arr // 16).astype(int)
        )))
        
        # Movement analysis
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        
        # Backtracking (negative x movement)
        backtrack_amount = np.sum(np.abs(dx[dx < 0]))
        forward_amount = np.sum(dx[dx > 0])
        
        # Vertical movement (jumping indicator)
        vertical_movement = np.sum(np.abs(dy))
        
        # Path length
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        
        # Average speed
        duration_ticks = len(traj)
        avg_speed = path_length / duration_ticks if duration_ticks > 0 else 0
        
        # State changes (power-up tracking)
        state_changes = np.sum(np.diff(states) != 0)
        
        record = {
            'trajectory_id': row['trajectory_id'],
            'vote_id': row.get('vote_id'),
            'level_id': row['level_id'],
            'player_id': row.get('player_id'),
            'side': row.get('side'),
            'max_x_reached': max_x,
            'unique_tiles_visited': unique_positions,
            'backtrack_amount': backtrack_amount,
            'forward_amount': forward_amount,
            'backtrack_ratio': backtrack_amount / (forward_amount + 1),
            'vertical_movement': vertical_movement,
            'path_length': path_length,
            'avg_speed': avg_speed,
            'duration_ticks': duration_ticks,
            'state_changes': state_changes,
            'final_state': states[-1] if states else 0,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} trajectory feature records")
    return df

def compute_generator_stats(level_stats_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics per generator."""
    
    # From level stats
    gen_level_stats = level_stats_df.groupby('generator_id').agg({
        'level_id': 'count',
        'times_shown': 'sum',
        'times_won': 'sum',
        'times_lost': 'sum',
        'times_tied': 'sum',
        'times_completed': 'sum',
        'total_deaths': 'sum',
        'total_play_time_seconds': 'sum',
        'win_rate': 'mean',
        'completion_rate': 'mean',
        'avg_deaths': 'mean',
        'difficulty_score': 'mean',
        'tag_fun': 'sum',
        'tag_boring': 'sum',
        'tag_too_hard': 'sum',
        'tag_too_easy': 'sum',
        'tag_creative': 'sum',
        'tag_good_flow': 'sum',
        'tag_unfair': 'sum',
        'tag_confusing': 'sum',
        'tag_impossible': 'sum',
    }).rename(columns={'level_id': 'num_levels'})
    
    # From telemetry
    if len(telemetry_df) > 0:
        gen_telemetry = telemetry_df.groupby('generator_id').agg({
            'vote_id': 'count',
            'won': 'sum',
            'duration_seconds': 'mean',
            'deaths': 'mean',
            'completed': 'mean',
            'coins_collected': 'mean',
            'enemies_stomped': 'mean',
            'jumps': 'mean',
        }).rename(columns={
            'vote_id': 'num_plays',
            'won': 'total_wins',
            'duration_seconds': 'avg_duration',
            'deaths': 'avg_deaths_telemetry',
            'completed': 'completion_rate_telemetry',
        })
        
        # Merge
        gen_stats = gen_level_stats.join(gen_telemetry, how='left')
    else:
        gen_stats = gen_level_stats
    
    gen_stats = gen_stats.reset_index()
    print(f"Computed stats for {len(gen_stats)} generators")
    return gen_stats

def compute_player_voting_patterns(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """Compute voting patterns per player."""
    if len(telemetry_df) == 0:
        return pd.DataFrame()
    
    # Only look at winning side telemetry
    winning_plays = telemetry_df[telemetry_df['won'] == True]
    
    player_stats = telemetry_df.groupby('player_id').agg({
        'vote_id': 'nunique',
        'won': 'sum',
        'deaths': 'mean',
        'duration_seconds': 'mean',
        'completed': 'mean',
        'jumps': 'mean',
    }).rename(columns={
        'vote_id': 'num_votes',
        'won': 'num_wins',
        'deaths': 'avg_deaths',
        'duration_seconds': 'avg_duration',
        'completed': 'completion_rate',
        'jumps': 'avg_jumps',
    })
    
    # Preferred difficulty (avg death rate of levels they voted for)
    if len(winning_plays) > 0:
        preferred_difficulty = winning_plays.groupby('player_id')['deaths'].mean()
        player_stats['preferred_difficulty'] = preferred_difficulty
    
    player_stats = player_stats.reset_index()
    print(f"Computed voting patterns for {len(player_stats)} players")
    return player_stats

def main():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("PCG Arena - Data Preparation")
    print("=" * 60)
    
    # Load raw data
    print("\n[1/6] Loading raw data...")
    level_stats_df = load_level_stats()
    votes_df = load_votes()
    trajectories_df = load_trajectories()
    player_profiles_df = load_player_profiles()
    
    # Extract features
    print("\n[2/6] Extracting telemetry features...")
    telemetry_df = extract_telemetry_features(votes_df)
    
    print("\n[3/6] Extracting trajectory features...")
    trajectory_features_df = extract_trajectory_features(trajectories_df)
    
    # Compute aggregates
    print("\n[4/6] Computing generator statistics...")
    generator_stats_df = compute_generator_stats(level_stats_df, telemetry_df)
    
    print("\n[5/6] Computing player voting patterns...")
    player_voting_df = compute_player_voting_patterns(telemetry_df)
    
    # Save processed data
    print("\n[6/6] Saving processed data...")
    
    level_stats_df.to_csv(OUTPUT_DIR / "level_stats_clean.csv", index=False)
    votes_df.to_json(OUTPUT_DIR / "votes_clean.json", orient='records', indent=2)
    telemetry_df.to_csv(OUTPUT_DIR / "telemetry_flat.csv", index=False)
    trajectory_features_df.to_csv(OUTPUT_DIR / "trajectory_features.csv", index=False)
    generator_stats_df.to_csv(OUTPUT_DIR / "generator_stats.csv", index=False)
    player_voting_df.to_csv(OUTPUT_DIR / "player_voting_patterns.csv", index=False)
    player_profiles_df.to_csv(OUTPUT_DIR / "player_profiles_clean.csv", index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Levels:       {len(level_stats_df):,}")
    print(f"Votes:        {len(votes_df):,}")
    print(f"Telemetry:    {len(telemetry_df):,} play records")
    print(f"Trajectories: {len(trajectory_features_df):,}")
    print(f"Generators:   {len(generator_stats_df):,}")
    print(f"Players:      {len(player_voting_df):,}")
    print("\nOutput files saved to:", OUTPUT_DIR)
    
    return {
        'level_stats': level_stats_df,
        'votes': votes_df,
        'telemetry': telemetry_df,
        'trajectory_features': trajectory_features_df,
        'generator_stats': generator_stats_df,
        'player_voting': player_voting_df,
        'player_profiles': player_profiles_df,
    }

if __name__ == "__main__":
    data = main()
