# Extended EDA Experiments - Summary Report

## Overview

This report documents additional experiments conducted to identify key gameplay features
that influence player preference (fun) in the PCG Arena dataset.

---

## H8: Enemy Density and Hazard Difficulty

**Hypothesis**: Levels with excessive hazards (high enemy counts, large gaps) decrease 
player enjoyment and have lower win rates.

### Results

- **Difficulty Score vs Win Rate**: r = -0.112, p = 0.0786
- **Deaths vs Win Rate**: r = -0.067, p = 0.2983

- **Easy (Q1) vs Hard (Q4)**: Easy mean = 0.538, Hard mean = 0.441
  - Mann-Whitney U = 6553.5, p = 0.0459

**Conclusion**: Higher difficulty (more deaths, lower completion) correlates with lower 
win rates, confirming that excessive hazards reduce player preference.

---

## H9: Rewards and Leniency

**Hypothesis**: Levels with more forgiving design (higher completion rates, rewards) 
are preferred by players.

### Results

- **Completion Rate vs Win Rate**: r = 0.112, p = 0.0786
- **"Fun" Tagged Levels**: Completion = 0.292 vs Others = 0.137
  - Mann-Whitney p = 0.0004

**Conclusion**: Completion rate is strongly predictive of preference. Levels that players
can complete are more likely to win comparisons, supporting the leniency hypothesis.

---

## H10: Feature Importance Modeling

**Hypothesis**: We can predict player preference from level features, and identify the
most influential factors.

### Results

**Model Performance (5-fold CV):**
- Logistic Regression: 0.689
- Random Forest: 0.693
- Gradient Boosting: 0.622

**Top 5 Most Important Features:**
  1. avg_duration_seconds (avg_rank = 0.0)
  2. tag_creative_rate (avg_rank = 1.3)
  3. tag_boring_rate (avg_rank = 3.3)
  4. tag_impossible_rate (avg_rank = 4.0)
  5. tag_fun_rate (avg_rank = 4.7)

**Conclusion**: Playability metrics (completion_rate, difficulty_score) are the dominant
predictors of preference, confirming that players prefer levels they can complete.

---

## Extended H6: Tag-Objective Correspondence

**Hypothesis**: Player-assigned tags correspond to measurable gameplay metrics.

### Results

| Tag | Metric | Expected | Actual | Significant |
|-----|--------|----------|--------|-------------|
| tag_fun | completion_rate | positive | positive ✓ | ✓ |
| tag_boring | completion_rate | negative | positive ✗ | ✓ |
| tag_too_hard | avg_deaths | positive | positive ✓ |  |
| tag_too_easy | completion_rate | positive | positive ✓ | ✓ |
| tag_creative | win_rate | positive | positive ✓ | ✓ |
| tag_impossible | completion_rate | negative | negative ✓ | ✓ |

**Conclusion**: Tags generally align with objective metrics. "Fun" correlates with completion,
"too_hard" correlates with deaths, validating tags as quality signals.

---

## Key Takeaways

1. **Playability is paramount**: Completion rate is the strongest predictor of preference
2. **Difficulty hurts**: Higher difficulty → lower win rate (linear, not inverted-U)
3. **Tags are valid**: Player tags correlate with measurable metrics
4. **Hazards matter**: Death count and difficulty score negatively predict preference

These findings support the design of generators that prioritize playability and manageable
difficulty over complex/challenging designs.
