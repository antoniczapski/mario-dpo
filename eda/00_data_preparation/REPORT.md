# Data Preparation Report

## Overview
This module loads and preprocesses the raw PCG Arena data exports for downstream analysis.

## Data Sources
| File | Records | Description |
|------|---------|-------------|
| `pcg-arena-level-stats-*.json` | 631 | Aggregated level performance metrics |
| `pcg-arena-votes-*.json` | 447 | Individual battle votes with telemetry |
| `pcg-arena-trajectories-*.json` | 100 | Player movement paths (sampled) |
| `pcg-arena-player-profiles-*.json` | 26 | Anonymous player profiles |

## Processed Outputs

### 1. `level_stats_clean.csv`
Clean level statistics with win rates, completion rates, tag counts, and difficulty scores.

### 2. `telemetry_flat.csv` (894 records)
Flattened telemetry from votes, one row per level-play:
- `vote_id`, `player_id`, `session_id`
- `side` (left/right), `level_id`, `generator_id`
- `won`, `lost`, `tied` - outcome indicators
- `duration_seconds`, `completed`, `deaths`
- `coins_collected`, `enemies_stomped`, `jumps`
- `death_by_enemy`, `death_by_fall`, `death_by_timeout`
- `avg_death_x`, `max_death_x` - spatial death features

### 3. `trajectory_features.csv` (100 records)
Extracted movement features:
- `max_x_reached` - furthest point reached
- `unique_tiles_visited` - spatial coverage
- `backtrack_amount`, `backtrack_ratio` - exploration behavior
- `vertical_movement` - jumping activity
- `path_length`, `avg_speed` - movement efficiency

### 4. `generator_stats.csv` (14 generators)
Aggregated generator-level metrics:
- Total levels, plays, wins
- Average win rate, completion rate
- Tag totals (fun, boring, too_hard, etc.)

### 5. `player_voting_patterns.csv` (26 players)
Per-player voting behavior:
- `num_votes`, `num_wins`
- `avg_deaths`, `avg_duration`, `completion_rate`
- `preferred_difficulty` - avg death rate of preferred levels

## Key Observations

1. **Limited Trajectory Data**: Only 100 trajectory records are available (likely due to export limit), which constrains some analyses.

2. **Active Generators**: 14 generators are currently active in the platform.

3. **Vote Distribution**: 447 votes × 2 sides = 894 individual level plays recorded.

## Data Quality Notes

- Many structural features in `level_stats` are `null` (e.g., `enemy_density`, `gap_count`) - these would need to be extracted from raw level files.
- Trajectory data is sampled, not complete for all plays.
- Player skill ratings are at default values (1000) suggesting the skill system may not be fully utilized.
