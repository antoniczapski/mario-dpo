# PCG Arena - Exploratory Data Analysis (EDA) Plan

This document outlines the plan for analyzing the PCG Arena dataset to understand procedural content generation quality, player preferences, and engagement metrics.

## 1. Dataset Overview

We have four primary data files available in this directory (`./eda/`):

*   **`pcg-arena-level-stats-*.json`**: Aggregated performance metrics per level (win rate, deaths, tags).
    *   *Note*: Many structural features (e.g., `enemy_density`, `gap_density`) appear to be `null` in the current export and may need to be computed from raw level files or inferred.
*   **`pcg-arena-votes-*.json`**: Individual battle records containing `player_id`, `generator_id`, `result` (outcome), and detailed `telemetry` (deaths, duration, events).
*   **`pcg-arena-trajectories-*.json`**: Raw player movement paths `(x, y, tick)` for each played level.
*   **`pcg-arena-player-profiles-*.json`**: Aggregated player statistics.

## 2. Prequisite Tasks: Data Preparation & Feature Extraction

Before testing hypotheses, the following preparation is required:

1.  **Metric Aggregation**: Merge `votes` data with `level_stats` to ensure we have the most granular interactions (e.g., did *this* specific player die in *this* specific way?).
2.  **Feature Extraction (Crucial)**:
    *   **Structural Features**: Since the JSON dump has `null` for fields like `enemy_density` or `gap_count`, we must write a script to parse the actual ASCII level files (available in the repo) and compute these metrics for every `level_id`.
        *   *Metrics to extract*: Enemy count, enemy type diversity, gap count, gap width, platform height variance, linearity (leniency).
    *   **Trajectory Features**: Process `trajectories` to compute "Path Diversity" and "Exploration" metrics.
        *   *Spatial Coverage*: \% of reachable tile space covered by the player.
        *   *Backtracking*: Amount of leftward movement (sum of negative dx).
        *   *Jumps*: Number of jumps (derived from y-velocity changes or telemetry events).
    *   **Difficulty Proxies**: Compute `Deaths per Minute` and `Win Rate` per level.

## 3. Research Questions & Hypotheses

### RQ1: What makes a "good" generator or level?
*Goal: Identify features that correlate with high Win Rate and "Fun" tags.*

*   **H1: Optimized Difficulty (Flow Channel)**
    *   **Hypothesis**: Levels with a "moderate" difficulty (e.g., 15-40% death rate) have higher win rates than levels with 0% or >60% death rates.
    *   **Test**: Plot `Win Rate` vs. `Death Rate` (quadratic regression). Check for an inverted U-shape.
    
*   **H2: Structural Variety**
    *   **Hypothesis**: Levels with higher variance in terrain height and more distinct enemy types are preferred over flat or repetitive levels.
    *   **Test**: Correlation analysis between `Win Rate` and `Height Variance` / `Enemy Entropy`.

*   **H3: Path Freedom (Agency)**
    *   **Hypothesis**: Players prefer levels that allow for more exploration (higher spatial coverage in trajectories) rather than strict linear paths.
    *   **Test**: Correlation between `Win Rate` and `Trajectory Spatial Coverage`.

### RQ2: Are players consistent in their preferences?
*Goal: Determine if "universal" quality exists or if we have distinct player clusters.*

*   **H4: Player Clusters (Bartle Types adaptation)**
    *   **Hypothesis**: We can cluster players into distinct groups (e.g., "Explorers" who prefer open levels vs. "Achievers" who prefer hard challenges).
    *   **Test**: Perform K-Means clustering on player vectors.
        *   *Vector features*: Avg. preferred difficulty (death rate of won levels), avg. duration of play, frequency of "Creative" vs. "Fun" tags.

*   **H5: Preference Consistency**
    *   **Hypothesis**: High-skill players (low death rate) are more consistent in their voting (lower variance in ratings for the same generator) than low-skill players.
    *   **Test**: Split players by skill quartile; compare the standard deviation of Elo updates or the agreement rate on "control" battles (if any).

### RQ3: What do tags actually mean?
*Goal: Validate user-submitted tags against objective telemetry.*

*   **H6: "Hard" Tag Validity**
    *   **Hypothesis**: The "Too Hard" tag strongly correlates with `Death Count` and `Duration`, but "Broken" correlates with `Zero Completion` + `Short Duration` (rage quit).
    *   **Test**: Logistic regression to predict tag presence based on telemetry metrics.

*   **H7: "Creative" vs. "Fun"**
    *   **Hypothesis**: Telescopic hypothesis - "Fun" correlates with Flow (H1), while "Creative" correlates with novel structural features (unseen patterns) regardless of difficulty.
    *   **Test**: Feature importance analysis (Random Forest) for predicting "Creative" vs. "Fun".

## 4. General EDA Tasks (Non-Hypothesis Driven)

Perform these exploratory steps to get a "feel" for the data:

1.  **Global Distributions**:
    *   Plot histograms of `Win Rate`, `Completion Rate`, and `Vote Count` per generator.
    *   Plot a heat map of `Death Locations` aggregated across all levels to see if deaths are concentrated at start/end.

2.  **Generator Fingerprinting**:
    *   Radar charts for top 5 generators comparing them on 5 axes: `Difficulty`, `Length`, `Verticality`, `Enemy Density`, `Playfulness` (jumps/minute).
  
3.  **Confusion Matrix**:
    *   Visualize the raw win-rate matrix between every pair of generators ($G_i$ vs $G_j$). Identify intransitive relationships (Rock-Paper-Scissors loops).

4.  **Trajectory Visualization**:
    *   Overlay top 10 "best" and bottom 10 "worst" level trajectories. Look for visual patterns (e.g., do bad levels have "traps" where everyone dies at $x=50$?).
