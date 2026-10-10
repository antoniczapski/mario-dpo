# PCG Arena EDA Report — March 23, 2026

## Dataset Summary

| Metric | Previous (Jan 28) | Current (Mar 23) | Growth |
|--------|-------------------|-------------------|--------|
| Levels | 748 | 1074 | 1.4x |
| Total Votes | 571 | 1109 | 1.9x |
| Players | 27 | 71 | 2.6x |
| Trajectories | 100 | 100 (fetched) | — |
| Telemetry records | ~1,142 | 2000 | 1.8x |

## Generator Rankings

| Rank | Generator | Win Rate | Games | Levels | Completion | Avg Deaths | Avg Duration |
|------|-----------|----------|-------|--------|------------|------------|-------------|
| 1 | test-gen | 100.0% | 1 | 1 | 0.0% | 1.00 | 3.5s |
| 2 | original | 79.3% | 164 | 14 | 22.0% | 0.78 | 22.9s |
| 3 | mariodpo | 72.7% | 110 | 86 | 25.0% | 0.74 | 22.4s |
| 4 | mariogpt | 59.1% | 159 | 83 | 27.3% | 0.67 | 23.3s |
| 5 | ore | 57.8% | 161 | 86 | 16.1% | 0.84 | 17.5s |
| 6 | notch | 47.9% | 144 | 71 | 21.5% | 0.78 | 11.2s |
| 7 | mariogan | 45.7% | 164 | 75 | 23.7% | 0.75 | 15.7s |
| 8 | hopper | 44.0% | 159 | 83 | 20.3% | 0.80 | 12.3s |
| 9 | genetic | 41.8% | 158 | 81 | 33.4% | 0.67 | 88.1s |
| 10 | marioDiffusion | 36.5% | 167 | 83 | 6.6% | 0.93 | 12.2s |
| 11 | notchParamRand | 28.7% | 160 | 86 | 35.9% | 0.64 | 12.8s |
| 12 | patternWeightCount | 27.2% | 173 | 87 | 6.0% | 0.91 | 13.1s |
| 13 | patternOccur | 21.9% | 155 | 79 | 4.9% | 0.88 | 12.8s |
| 14 | notchParam | 19.3% | 150 | 82 | 12.5% | 0.87 | 15.4s |
| 15 | patternCount | 14.3% | 147 | 77 | 4.7% | 0.92 | 10.2s |

## RQ1: What Makes a Good Level?

### H1: Difficulty vs Win Rate
- **Result**: Spearman r = -0.194, p = 0.0015 (n = 265)
- **Interpretation**: Significant negative correlation — easier levels win more.
- **Flow channel hypothesis**: Not supported. Monotonic negative relationship, not inverted-U.

### H2: Tag Correlations with Win Rate
| Tag | Spearman r | p-value | Significant |
|-----|-----------|---------|-------------|
| fun | 0.237 | 0.0001 | ✅ |
| creative | 0.185 | 0.0025 | ✅ |
| too_hard | -0.179 | 0.0034 | ✅ |
| boring | -0.147 | 0.0168 | ✅ |
| unfair | -0.139 | 0.0238 | ✅ |
| confusing | -0.113 | 0.0659 | ❌ |
| good_flow | 0.091 | 0.1378 | ❌ |
| too_easy | 0.045 | 0.4657 | ❌ |

### H3: Telemetry — Won vs Lost Plays
| Metric | Won Mean | Lost Mean | p-value | Significant |
|--------|----------|-----------|---------|-------------|
| duration_seconds | 21.402 | 10.769 | 0.0000 | ✅ |
| deaths | 0.744 | 0.822 | 0.0002 | ✅ |
| completed | 0.249 | 0.144 | 0.0000 | ✅ |
| coins_collected | 1.861 | 0.468 | 0.0000 | ✅ |
| jumps | 16.196 | 8.505 | 0.0000 | ✅ |

### Completion Rate vs Win Rate
- Spearman r = 0.227, p = 0.0002

## RQ3: Tag Validation (H6)

| Tag → Metric | Tagged Mean | Not Tagged Mean | Tagged n | p-value | Significant |
|-------------|-------------|-----------------|----------|---------|-------------|
| Fun → Completion | 0.500 | 0.177 | 50 | 0.0000 | ✅ |
| Too Hard → Deaths | 0.929 | 0.792 | 56 | 0.0125 | ✅ |
| Too Easy → Completion | 0.789 | 0.180 | 19 | 0.0000 | ✅ |
| Creative → Wins | 0.925 | 0.398 | 40 | 0.0000 | ✅ |
| Boring → Duration | 20.698 | 21.370 | 51 | 0.0016 | ✅ |

## Player Engagement

| Metric | Value |
|--------|-------|
| Total registered players | 71 |
| Active players (≥1 vote) | 71 |
| Median votes per player | 6 |
| Mean votes per player | 15.6 |
| Most active player | 290 votes |
| Players contributing 80% of votes | 21 |

## Tag Distribution

| Tag | Count |
|-----|-------|
| too_hard | 56 |
| boring | 51 |
| fun | 50 |
| creative | 40 |
| too_easy | 19 |
| good_flow | 0 |
| unfair | 0 |
| confusing | 0 |

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
