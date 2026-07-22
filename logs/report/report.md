# Experiment Report

## 1. Experiment Overview

- **Name:** Legacy Experiment
- **Timestamp:** 20260721_224007
- **Type:** comparison
- **Base Seed:** 42 (30 seeds)

## 2. Configuration

| Config Key | Description |
|---|---|
| **A** | Shortest-Path Baseline |
| **B** | Risk-Aware A* |
| **C** | Conservative Risk-Aware |
| **D** | Risk-Aware (No Collision Avoidance) |
| **E** | Risk-Aware (Full, Collision Enabled) |

## 3. Overall Performance

- **Overall Success Rate:** 59.2%
- **Overall Efficiency:** 1.601
- **Mine Death Rate:** 27.4%
- **Collision Rate:** 0.0%
- **Timeout Rate:** 13.4%

## 4. Robot Performance

| Robot | Success Rate | Efficiency | Replans | Waits | Mine Death Rate |
|---|---|---|---|---|---|
| Blue | 60.7% | 1.564 | 5.629 | 3.506 | 26.9% |
| Green | 58.5% | 1.650 | 5.686 | 3.451 | 27.9% |
| Red | 60.0% | 1.583 | 5.503 | 3.224 | 27.0% |
| Yellow | 57.5% | 1.607 | 5.819 | 3.564 | 27.9% |

## 5. Configuration Comparison

| Config | Success Rate | Efficiency | Replans | Waits | Mine Death Rate |
|---|---|---|---|---|---|
| A | 59.4% | 1.669 | 2.923 | 0.018 | 37.0% |
| B | 58.1% | 1.645 | 6.249 | 4.168 | 26.0% |
| C | 59.7% | 1.542 | 6.192 | 4.132 | 24.3% |
| D | 59.6% | 1.587 | 6.481 | 4.488 | 24.9% |
| E | 59.2% | 1.556 | 6.452 | 4.376 | 24.9% |

## 6. Key Metrics

Below are the visualised key metrics from the experiment:

![01 Avg Wins By Robot](figures/01_avg_wins_by_robot.png)
![02 Survival Rate By Robot](figures/02_survival_rate_by_robot.png)
![03 Efficiency By Robot](figures/03_efficiency_by_robot.png)
![04 Mine Death Rate](figures/04_mine_death_rate.png)
![05 Collision Death Rate](figures/05_collision_death_rate.png)
![06 Timeout Rate](figures/06_timeout_rate.png)
![07 Waiting Behavior](figures/07_waiting_behavior.png)
![08 Speed Vs Efficiency](figures/08_speed_vs_efficiency.png)
![09 Speed Vs Survival](figures/09_speed_vs_survival.png)
![10 Risk Vs Mine Deaths](figures/10_risk_vs_mine_deaths.png)
![11 Risk Vs Efficiency](figures/11_risk_vs_efficiency.png)
![12 Patience Vs Waiting](figures/12_patience_vs_waiting.png)
![13 Patience Vs Survival](figures/13_patience_vs_survival.png)
![14 Risk Vs Survival](figures/14_risk_vs_survival.png)
![15 Efficiency Distribution](figures/15_efficiency_distribution.png)
![16 Survival Distribution](figures/16_survival_distribution.png)
![17 Efficiency Boxplot](figures/17_efficiency_boxplot.png)
![18 Survival Boxplot](figures/18_survival_boxplot.png)
![19 Tournament Correlation Matrix](figures/19_tournament_correlation_matrix.png)
![20 Efficiency Across Seeds](figures/20_efficiency_across_seeds.png)
![21 Win Rate Across Seeds](figures/21_win_rate_across_seeds.png)
![22 Death Composition](figures/22_death_composition.png)
![23 Speed Distribution](figures/23_speed_distribution.png)
![24 Risk Distribution](figures/24_risk_distribution.png)
![25 Patience Distribution](figures/25_patience_distribution.png)
![26 Efficiency Survival Tradeoff](figures/26_efficiency_survival_tradeoff.png)
![27 Composite Performance Score](figures/27_composite_performance_score.png)
![28 Config Success Rate](figures/28_config_success_rate.png)
![29 Config Efficiency](figures/29_config_efficiency.png)
![30 Config Mine Death Rate](figures/30_config_mine_death_rate.png)
![31 Config Collision Rate](figures/31_config_collision_rate.png)
![32 Config Timeout Rate](figures/32_config_timeout_rate.png)
![33 Config Replanning](figures/33_config_replanning.png)
![34 Config Waiting](figures/34_config_waiting.png)
![35 Unknown Cells Entered](figures/35_unknown_cells_entered.png)
![36 Excess Steps](figures/36_excess_steps.png)
![37 Efficiency Config Boxplot](figures/37_efficiency_config_boxplot.png)
![38 Replans Vs Efficiency](figures/38_replans_vs_efficiency.png)
![39 Waits Vs Efficiency](figures/39_waits_vs_efficiency.png)
![40 Unknown Cells Vs Efficiency](figures/40_unknown_cells_vs_efficiency.png)
![41 Robot Efficiency Experiment](figures/41_robot_efficiency_experiment.png)
![42 Robot Success Experiment](figures/42_robot_success_experiment.png)
![43 Config Robot Efficiency Heatmap](figures/43_config_robot_efficiency_heatmap.png)
![44 Config Robot Success Heatmap](figures/44_config_robot_success_heatmap.png)
![45 Efficiency Across Rounds](figures/45_efficiency_across_rounds.png)
![46 Success Across Rounds](figures/46_success_across_rounds.png)
![47 Replanning Distribution](figures/47_replanning_distribution.png)
![48 Wait Distribution](figures/48_wait_distribution.png)
![49 Excess Step Distribution](figures/49_excess_step_distribution.png)
![50 Experiment Correlation Matrix](figures/50_experiment_correlation_matrix.png)
![51 Configuration Overall Score](figures/51_configuration_overall_score.png)
![52 Robot Radar Chart](figures/52_robot_radar_chart.png)
![53 Failure Reasons](figures/53_failure_reasons.png)
![54 Efficiency By Robot Boxplot](figures/54_efficiency_by_robot_boxplot.png)
![55 Replans Vs Efficiency Regression](figures/55_replans_vs_efficiency_regression.png)

## 7. Failure Analysis

| Config | Mine Death Rate | Collision Rate | Timeout Rate |
|---|---|---|---|
| A | 37.0% | 0.0% | 3.7% |
| B | 26.0% | 0.0% | 15.9% |
| C | 24.3% | 0.0% | 16.0% |
| D | 24.9% | 0.0% | 15.5% |
| E | 24.9% | 0.0% | 15.9% |

## 8. Interesting Findings

- **Best Performing Config:** C with success rate 59.7%.
- **Worst Performing Config:** B with success rate 58.1%.
- **Most Efficient Robot:** Green with efficiency 1.650.
- **Least Efficient Robot:** Blue with efficiency 1.564.

## 9. Performance Bottlenecks

- **Highest Wait Time:** Config D averages 4.488 waits per run.
- **Highest Replanning Overhead:** Config D averages 6.481 replans per run.

## 10. Planner Behaviour

Configurations exhibit distinct behavioural patterns. Higher replan counts often indicate dynamic pathfinding in response to unexpected obstacles, whereas high wait times suggest collision-avoidance mechanisms taking precedence over movement.

## 11. Suggested Improvements

Based on the data collected, consider the following improvements:
1. **Tune Collision Avoidance:** Adjust wait priorities and fallback paths to reduce excessive idling when robots meet.
2. **Optimise Replanning Thresholds:** Introduce a caching layer or replan-delay to prevent computational overhead in densely clustered obstacle regions.
3. **Enhance Risk Modelling:** Adjust the risk-aware cost heuristic to balance safety against path efficiency, potentially reducing mine deaths without overly extending paths.

## Important Interpretation Warnings

- A single seed is not enough to claim that one planner is better.
- A high completion rate can hide terrible efficiency or excessive waiting.
- Survival rate and completion rate are different metrics.
- Mine deaths and collision deaths should be analyzed separately.
- Replanning can indicate intelligence, but excessive replanning can also indicate instability.
- Oracle efficiency is only meaningful if the oracle path is valid under the same movement constraints.