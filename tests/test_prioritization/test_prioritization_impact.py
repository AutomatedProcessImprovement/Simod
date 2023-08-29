config_yaml_A = """
version: 2
common:
  log_path: tests/assets/___.csv
  log_ids:
    case: case_id
    activity: Activity
    resource: Resource
    start_time: start_time
    end_time: end_time
  repetitions: 10
  evaluation_metrics: 
    - absolute_hourly_emd
    - cycle_time_emd
    - circadian_emd
preprocessing:
  multitasking: false
control_flow:
  optimization_metric: cycle_time_emd
  max_evaluations: 10
  mining_algorithm: sm1
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gateway_probabilities:
    - discovery
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
resource_model:
  optimization_metric: absolute_hourly_emd
  max_evaluations: 10
  discover_prioritization_rules: true
  resource_profiles:
    discovery_type: differentiated
    granularity: 
      - 15
      - 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
extraneous_activity_delays:
  optimization_metric: absolute_emd
  num_iterations: 1
"""

# @pytest.mark.manual
# def test_prioritization_discovery_impact(entry_point: Path):
#     settings = SimodSettings.from_stream(config_yaml_A)
#     settings.log_path = (entry_point / Path(settings.common.log_path.name)).absolute()
#
#     log = EventLog.from_path(
#         path=settings.common.log_path,
#         log_ids=settings.common.log_ids,
#         process_name=settings.common.log_path.stem,
#     )
#
#     simod = Simod(settings=settings, event_log=log)
#     simod.run()

# TODO: find a log with prioritization _rules
