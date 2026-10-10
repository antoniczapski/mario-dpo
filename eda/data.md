# PCG Arena – Data Description

**Export date:** 2026-05-10
**Source directory:** `eda/data_10_05_2026/`
**Protocol version:** `arena/v0`

The dataset consists of four JSON files, all following the same envelope schema:

```json
{
  "protocol_version": "arena/v0",
  "export_type": "<type>",
  "total": <int>,
  "limit": <int|null>,
  "offset": <int|null>,
  "filter": <object|null>,
  "data": [...]
}
```

---

## 1. Player Profiles (`pcg-arena-player-profiles-2026-05-10.json`)

### Summary

| Field                       | Value                      |
|-----------------------------|----------------------------|
| Anonymous player profiles   | **79**                     |
| Linked users                | 0 (all anonymous)          |
| Active period               | 2026-01-01 – 2026-05-03    |

### Schema

| Field                    | Type          | Description |
|--------------------------|---------------|-------------|
| `player_id`              | string        | Browser-based anonymous player identifier (`anon_<uuid>`); not a verified unique human |
| `first_seen_utc`         | datetime      | Timestamp of first session |
| `last_seen_utc`          | datetime      | Timestamp of most recent session |
| `total_battles`          | int           | Number of battles participated in |
| `total_votes`            | int           | Number of votes cast |
| `total_sessions`         | int           | Number of play sessions |
| `avg_battles_per_session`| float \| null | Average battles per session |
| `skill_rating`           | float         | Glicko-style skill estimate (default 1000) |
| `skill_rd`               | float         | Rating deviation (default 350 = unrated) |
| `prefers_harder_count`   | int           | Times player chose the harder level |
| `prefers_easier_count`   | int           | Times player chose the easier level |
| `prefers_longer_count`   | int           | Times player chose the longer level |
| `prefers_shorter_count`  | int           | Times player chose the shorter level |
| `linked_user_id`         | string \| null| Registered account ID if linked |

### Basic EDA

- **Total votes cast (across all anonymous player IDs):** 1–306 per player ID; mean ≈ 15.1, median = 6
- **Skill ratings:** All at default 1000 (rating system not yet converged / not exported post-update)
- **`total_battles`:** All 0 — battles field appears unused in the current schema version
- **Identity caveat:** Session continuity is based on browser/session identifiers. Clearing cookies or switching browser/device can create a new anonymous player ID.

---

## 2. Level Stats (`pcg-arena-level-stats-2026-05-10.json`)

### Summary

| Field           | Value   |
|-----------------|---------|
| Total levels    | **1104** |
| Generators      | 15      |
| Tile features   | 0 (not yet computed) |

### Generators and level counts

| Generator          | Levels |
|--------------------|--------|
| `patternWeightCount` | 90   |
| `mariodpo`          | 90    |
| `ore`               | 88    |
| `notchParamRand`    | 87    |
| `hopper`            | 85    |
| `notchParam`        | 85    |
| `marioDiffusion`    | 84    |
| `mariogpt`          | 84    |
| `genetic`           | 82    |
| `patternOccur`      | 81    |
| `mariogan`          | 79    |
| `patternCount`      | 77    |
| `notch`             | 77    |
| `original`          | 14    |
| `test-gen`          | 1     |

### Schema

**Battle outcome fields**

| Field                   | Type  | Description |
|-------------------------|-------|-------------|
| `level_id`              | string | `<generator_id>::<filename>` |
| `generator_id`          | string | Generator that produced this level |
| `times_shown`           | int   | Times presented in a battle |
| `times_won`             | int   | Times selected as winner |
| `times_lost`            | int   | Times the opponent won |
| `times_tied`            | int   | Times result was a tie |
| `times_skipped`         | int   | Times the battle was skipped |
| `times_play_skipped`    | int   | Times gameplay was skipped before voting |
| `times_completed`       | int   | Times level was completed (reached the flag) |
| `total_deaths`          | int   | Total one-attempt death/failure indicators across plays |
| `total_play_time_seconds`| float | Cumulative play time |
| `win_rate`              | float | `times_won / times_shown` |
| `completion_rate`       | float | `times_completed / times_shown` |
| `avg_deaths`            | float | Death/failure rate under the current one-attempt-per-side design |
| `avg_duration_seconds`  | float | Mean play duration |
| `difficulty_score`      | float | Derived difficulty estimate |
| `updated_at_utc`        | datetime | Last update timestamp |
| `computed_at_utc`       | datetime \| null | When tile features were computed |

**Tag fields** (integer counts of how many players applied each tag)

`tag_fun`, `tag_boring`, `tag_too_hard`, `tag_too_easy`, `tag_creative`, `tag_good_flow`, `tag_unfair`, `tag_confusing`, `tag_not_mario_like`, `tag_impossible`, `tag_broken_graphics`

**Tile-level structural features** (all `null` in current export — not yet computed)

`width`, `height`, `ground_tiles`, `platform_tiles`, `pipe_tiles`, `coin_tiles`, `question_block_tiles`, `brick_tiles`, `empty_tiles`, `enemy_goomba`, `enemy_koopa_red`, `enemy_koopa_green`, `enemy_spiky`, `enemy_piranha`, `enemy_bullet_bill`, `enemy_total`, `gap_count`, `max_gap_width`, `platform_count`, `avg_platform_height`, `height_variance`, `enemy_density`, `coin_density`, `gap_density`, `structural_complexity`, `leniency_score`

### Basic EDA

| Metric             | Min  | Max  | Mean | Median |
|--------------------|------|------|------|--------|
| `win_rate`         | 0.00 | 1.00 | 0.48 | 0.50   |
| `completion_rate`  | 0.00 | 1.00 | 0.19 | 0.00   |
| `avg_deaths`       | 0.00 | 1.00 | 0.80 | 1.00   |
| `times_shown`      | 1    | 18   | 2.16 | 2.00   |

- Win rates are broadly centred around 0.5 — consistent with balanced matchmaking.
- Median completion rate is 0 — most levels are never fully completed.
- `avg_deaths` should be interpreted as a one-attempt failure/death rate, not repeated deaths per level.

---

## 3. Votes (`pcg-arena-votes-2026-05-10.json`)

> **Note:** The file contains the first 1000 of **1192** total votes (limit=1000).

### Summary

| Field         | Value                    |
|---------------|--------------------------|
| Total votes   | **1192** (1000 in file)  |
| Date range    | 2026-01-09 – 2026-05-03  |

### Schema

| Field               | Type          | Description |
|---------------------|---------------|-------------|
| `vote_id`           | string        | Unique vote identifier (`v_<uuid>`) |
| `battle_id`         | string        | Battle this vote belongs to (`btl_<uuid>`) |
| `session_id`        | string        | Play session UUID |
| `player_id`         | string        | Anonymous browser/player ID that cast this vote |
| `created_at_utc`    | datetime      | When the vote was cast |
| `result`            | string        | `LEFT`, `RIGHT`, `TIE`, or `SKIP` |
| `left_generator_id` | string        | Generator of the left level |
| `right_generator_id`| string        | Generator of the right level |
| `left_level_id`     | string        | Level ID of the left level |
| `right_level_id`    | string        | Level ID of the right level |
| `left_tags`         | string[]      | Tags applied to the left level |
| `right_tags`        | string[]      | Tags applied to the right level |
| `telemetry`         | object        | Per-side gameplay telemetry (see below) |

**Telemetry sub-schema** (same structure for `telemetry.left` and `telemetry.right`)

| Field                  | Type     | Description |
|------------------------|----------|-------------|
| `level_id`             | string   | Level played |
| `played`               | bool     | Whether the level was actually played |
| `skipped`              | bool     | Whether gameplay was skipped |
| `completed`            | bool     | Whether the player reached the flag |
| `duration_seconds`     | float    | Seconds spent in this level |
| `deaths`               | int      | One-attempt death indicator/count (currently 0 or 1 per side) |
| `death_locations`      | object[] | `{cause, tick, x, y}` per death/failure location |
| `jumps`                | int      | Total jumps made |
| `coins_collected`      | int      | Coins collected |
| `lives_collected`      | int      | 1-UP mushrooms collected |
| `powerups_collected`   | int      | Total power-ups collected |
| `powerups_mushroom`    | int      | Mushrooms collected |
| `powerups_flower`      | int      | Fire flowers collected |
| `enemies_killed`       | int      | Total enemies killed |
| `enemies_stomped`      | int      | Enemies killed by stomping |
| `enemies_shell_killed` | int      | Enemies killed via shell |
| `enemies_fire_killed`  | int      | Enemies killed by fireball |
| `events`               | object[] | Ordered gameplay events (see Trajectories) |
| `trajectory`           | object[] | Positional samples `{tick, x, y, state}` |

### Basic EDA

**Vote results (from 1000 sampled votes)**

| Result | Count | % |
|--------|-------|---|
| LEFT   | 418   | 41.8% |
| RIGHT  | 398   | 39.8% |
| TIE    | 165   | 16.5% |
| SKIP   | 19    | 1.9% |

**Generator appearances in battles** (from 1000 sampled votes, each battle contributes 2)

| Generator          | Appearances |
|--------------------|-------------|
| `mariogan`         | 160 |
| `patternWeightCount`| 153 |
| `marioDiffusion`   | 152 |
| `ore`              | 149 |
| `hopper`           | 147 |
| `original`         | 146 |
| `notchParam`       | 145 |
| `genetic`          | 145 |
| `patternOccur`     | 141 |
| `notch`            | 141 |
| `notchParamRand`   | 138 |
| `mariogpt`         | 138 |
| `patternCount`     | 129 |
| `mariodpo`         | 116 |

**Tag frequency** (from 1000 sampled votes)

| Tag               | Count |
|-------------------|-------|
| `impossible`      | 87 |
| `fun`             | 56 |
| `too_hard`        | 55 |
| `boring`          | 49 |
| `creative`        | 39 |
| `broken_graphics` | 22 |
| `too_easy`        | 21 |

---

## 4. Trajectories (`pcg-arena-trajectories-2026-05-10.json`)

> **Note:** The file contains 100 of **2,429** total trajectories (limit=100). Each vote generates up to 2 trajectories (one per side).

The standalone trajectory export is paginated/truncated. For the replanned EDA, the primary trajectory source is the `telemetry.left.trajectory` and `telemetry.right.trajectory` arrays embedded in the vote export, which contain 1968 non-empty trajectories in the first 1000 vote records.

### Summary

| Field              | Value                  |
|--------------------|------------------------|
| Total trajectories | **2,429** (100 in file)|
| Side balance       | 51 left / 49 right     |

### Schema

| Field             | Type     | Description |
|-------------------|----------|-------------|
| `trajectory_id`   | string   | Unique ID (`traj_<uuid>`) |
| `vote_id`         | string   | Associated vote |
| `level_id`        | string   | Level this trajectory is from |
| `session_id`      | string   | Play session UUID |
| `player_id`       | string   | Player who generated this trajectory |
| `side`            | string   | `left` or `right` |
| `trajectory`      | object[] | Positional samples: `{tick, x, y, state}` |
| `death_locations` | object[] | `{cause, tick, x, y}` per death |
| `events`          | object[] | Gameplay events (see below) |
| `summary`         | object   | Aggregated summary (see below) |
| `created_at_utc`  | datetime | When recorded |

**Trajectory point schema:** `{tick: int, x: float, y: float, state: int}`
Sampled every 8 game ticks (~8 frames).

**Summary sub-schema**

| Field           | Type  | Description |
|-----------------|-------|-------------|
| `duration_ticks`| int   | Total ticks until death/win/end |
| `max_x_reached` | float | Furthest horizontal position reached |
| `death_count`   | int   | One-attempt death/failure count |
| `completed`     | bool  | Whether the level was completed |

**Event types and counts** (from 100 sampled trajectories)

| Event type   | Count |
|--------------|-------|
| `JUMP`       | 1031 |
| `LAND`       | 985  |
| `FALL_KILL`  | 303  |
| `BUMP`       | 85   |
| `LOSE`       | 82   |
| `STOMP_KILL` | 46   |
| `COLLECT`    | 36   |
| `HURT`       | 35   |
| `WIN`        | 18   |
| `KICK`       | 4    |
| `SHELL_KILL` | 2    |

### Basic EDA

| Metric                   | Min | Max | Mean  | Median |
|--------------------------|-----|-----|-------|--------|
| Trajectory length (ticks)| 3   | 297 | 44.1  | 28.0   |
| Events per trajectory    | 1   | 189 | 26.3  | 15.5   |

- Trajectories are short on average (~28 ticks ≈ 4 seconds at 60 fps / 8-tick sampling), consistent with early deaths.
- JUMP and LAND are by far the most common events; WIN is rare relative to LOSE.

---

## Cross-dataset Links

```
player_profiles  ──player_id──▶  votes  ──vote_id──▶  trajectories
                                   │
                               level_id
                                   │
                                   ▼
                            level_stats
```

- `votes.left_level_id` / `votes.right_level_id` → `level_stats.level_id`
- `votes.player_id` → `player_profiles.player_id`
- `trajectories.vote_id` → `votes.vote_id`
- `trajectories.player_id` → `player_profiles.player_id`

---

## Caveats

- **Votes file is truncated** to 1000 of 1192 records. The remaining 192 votes need a second request with `offset=1000`.
- **Trajectories file is truncated** to 100 of 2429 records.
- **Tile-level structural features** in `level_stats` are all `null` — not yet computed for this export.
- **Skill ratings** are all at the Glicko default (1000, RD=350), meaning the rating system output for this batch has not been persisted yet.
- `original` generator has only 14 levels (original SMB hand-crafted levels used as baseline).

---

## Analysis-ready derived tables

The replanned EDA pipeline writes derived outputs to `eda/07_replanned_analysis/outputs/`:

- `vote_table.csv` — one row per exported vote.
- `side_level_table.csv` — one row per vote side, including generator IDs, side outcome, score (`win=1`, `tie=0.5`, `loss=0`, `skip=missing`), telemetry summaries, tag flags, and trajectory summaries.
- `generator_ranking.csv` — generator score rates, bootstrap intervals, Bradley–Terry display ratings, completion rates, and one-attempt death/failure rates.
- `level_static_metrics.csv` and `generator_static_metrics.csv` — static expressive metrics computed from ASCII level text files rather than null `level_stats` structural columns.
- `generator_trajectory_metrics.csv` — trajectory occupancy, progress, verticality, path-diversity, and failure-location concentration summaries by generator.
