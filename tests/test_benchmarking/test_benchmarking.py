import pytest

from simod.configuration import Configuration
from simod.optimization.optimizer import Optimizer

config_str = """
version: 2
common:
  log_path: logs/confidential_1000_processed.xes
  exec_mode: optimizer
  repetitions: 1
  evaluation_metrics: 
    - dl
    - absolute_hourly_emd
  log_ids:
    case: "case:concept:name"
    activity: "concept:name"
    resource: "org:resource"
    start_time: "start_timestamp"
    end_time: "time:timestamp"
preprocessing:
  multitasking: false
structure:
  max_evaluations: 1
  mining_algorithm: sm3
  concurrency:
    - 0.0
    - 1.0
  epsilon:
    - 0.0
    - 1.0
  eta:
    - 0.0
    - 1.0
  gateway_probabilities:
    - equiprobable
    - discovery
  replace_or_joins:
    - true
    - false
  prioritize_parallelism:
    - true
    - false
calendars:
  max_evaluations: 1
  case_arrival:
    discovery_type: undifferentiated
    granularity: 60
    confidence:
      - 0.01
      - 0.1
    support:
      - 0.01
      - 0.1
    participation: 0.4
  resource_profiles:
    discovery_type: differentiated
    granularity: 60
    confidence:
      - 0.5
      - 0.85
    support:
      - 0.01 
      - 0.3
    participation: 0.4
"""

test_cases = [
    {
        'name': 'confidential_1000_processed',
        'log_name': 'confidential_1000_processed.xes',
        'config': config_str,
    },
    # {
    #     'log_name': 'confidential_2000_processed.xes',
    #     'name': 'confidential_2000_processed',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'BPI_Challenge_2012_W_Two_TS.xes',
    #     'name': 'BPI_Challenge_2012_W_Two_TS',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'ConsultaDataMining201618.xes',
    #     'name': 'ConsultaDataMining201618',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'Production.xes',
    #     'name': 'Production',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'PurchasingExample.xes',
    #     'name': 'PurchasingExample',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'cvs_pharmacy_processed.xes',
    #     'name': 'cvs_pharmacy_processed',
    #     'config': config_str
    # },

    # {
    #     'log_name': 'insurance.xes',
    #     'name': 'insurance',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'BPI_Challenge_2017_W_Two_TS.xes',
    #     'name': 'BPI_Challenge_2017_W_Two_TS',
    #     'config': config_str
    # },
    # {
    #     'log_name': 'Application-to-Approval-Government-Agency.xes',
    #     'name': 'Application-to-Approval-Government-Agency',
    #     'config': config_str
    # },
]


# @pytest.mark.benchmark
# @pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
# def test_benchmarking(test_data, entry_point):
#     settings: Configuration = Configuration.from_stream(test_data['config'])
#     settings.common.log_path = entry_point.absolute().parent / 'test_benchmarking/logs' / test_data['log_name']
#     optimizer = Optimizer(settings)
#
#     optimizer.run()
