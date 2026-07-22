import sys
sys.path.insert(0, '.')
errors = []

try:
    from analysis.models import RobotResult, RunResult, ExperimentMeta, ROBOT_NAMES, ROBOT_COLOURS
    print('[OK] analysis.models')
except Exception as e:
    errors.append(str(e)); print(f'[FAIL] models: {e}')

try:
    from analysis.loader import Experiment, detect_format
    print('[OK] analysis.loader')
except Exception as e:
    errors.append(str(e)); print(f'[FAIL] loader: {e}')

try:
    from analysis.metrics import compute_stats, runs_to_dataframe, aggregate_overall, aggregate_by_config, aggregate_by_robot, build_summary_dict, config_scorecard
    print('[OK] analysis.metrics')
except Exception as e:
    errors.append(str(e)); print(f'[FAIL] metrics: {e}')

try:
    from analysis.plots import plot_all, savefig, plot_bar
    print('[OK] analysis.plots')
except Exception as e:
    errors.append(str(e)); print(f'[FAIL] plots: {e}')

try:
    from analysis.report import generate_markdown_report, generate_html_report, write_all_reports
    print('[OK] analysis.report')
except Exception as e:
    errors.append(str(e)); print(f'[FAIL] report: {e}')

try:
    from analysis.main import main
    print('[OK] analysis.main')
except Exception as e:
    errors.append(str(e)); print(f'[FAIL] main: {e}')

# Round-trip test
if not errors:
    robots = [
        RobotResult('Blue',  True,  False, False, False, 20, 15, 3, 2, 5),
        RobotResult('Red',   False, True,  False, False, 0,  12, 1, 0, 3),
        RobotResult('Green', True,  False, False, False, 18, 14, 2, 1, 4),
        RobotResult('Yellow',True,  False, False, False, 25, 18, 5, 4, 8),
    ]
    run = RunResult(run_id=1, config='A', seed=42, ticks=87, world_mines=8, world_walls=33, robots=robots)
    runs = [run]

    df = runs_to_dataframe(runs)
    assert len(df) == 4, "Expected 4 robot rows"
    assert 'robot_name' in df.columns, "Missing robot_name column"
    print(f'[OK] runs_to_dataframe: {len(df)} rows, cols={list(df.columns)[:6]}...')

    stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert 'p95' in stats, "Missing p95 key"
    mean_val = stats['mean']
    print(f'[OK] compute_stats: mean={mean_val}')

    overall = aggregate_overall(runs)
    sr = overall['success_rate']['mean']
    print(f'[OK] aggregate_overall: success_rate={sr:.2f}')

    by_cfg = aggregate_by_config(runs)
    assert 'A' in by_cfg
    print(f'[OK] aggregate_by_config: configs={list(by_cfg.keys())}')

    by_robot = aggregate_by_robot(runs)
    assert 'Blue' in by_robot
    print(f'[OK] aggregate_by_robot: robots={list(by_robot.keys())}')

    sc = config_scorecard(runs)
    assert 'overall_score' in sc.columns
    score = sc.iloc[0]['overall_score']
    print(f'[OK] config_scorecard: score={score:.3f}')

print()
if errors:
    print(f'FAILED with {len(errors)} import errors')
    sys.exit(1)
else:
    print('ALL CHECKS PASSED')
