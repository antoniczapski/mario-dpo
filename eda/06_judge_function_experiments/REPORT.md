# Judge Function Experiments - Summary Report
Generated: 2026-02-01 18:52:04.155581

## Experiment Results Summary

### Experiment A: Verticality Validation
- Y-Sigma vs Win Rate: r = 0.257, p = 0.0190
- Path Entropy vs Win Rate: r = 0.275
- Hesitation vs Win Rate: r = -0.045
- Partial Correlation (controlling completion): r = 0.2770347486504337

### Experiment B: Hazard Hierarchy
- Data limitation: gap_density/enemy_density NULL in database
- Pattern generator win rate: 0.257
- Other generator win rate: 0.542

### Experiment C: Death Entropy
- Death Entropy vs Win Rate: r = nan, p = nan
- Early Death Rate vs Win Rate: r = -0.134
- Low entropy win rate: nan
- High entropy win rate: 0.460

### Experiment D: Original Centroid
- Mahalanobis Distance vs Win Rate: r = -0.279, p = 0.0105
- Style Reward vs Win Rate: r = 0.279

## Proposed Judge Function

```
Stage 1 (Static Gatekeeper):
  J_static = w_style * (1/(1+D_M)) - w_gap * GapDensity - w_early * EarlyHazards

Stage 2 (Simulation Judge):
  J_final = J_static + w_vert * σ_y + w_flow * (1-Hesitation) - w_choke * (1-DeathEntropy)
```
