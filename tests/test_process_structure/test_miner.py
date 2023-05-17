import tempfile
from pathlib import Path

import pytest
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod

from simod.control_flow.discovery import discover_process_model
from simod.control_flow.settings import HyperoptIterationParams
from simod.settings.common_settings import Metric
from simod.settings.control_flow_settings import ProcessModelDiscoveryAlgorithm

structure_config_sm2 = {
    "mining_algorithm": "sm2",
    "max_eval_s": 2,
    "concurrency": 0.5,
    "epsilon": 0.15,
    "eta": 0.87,
    "gate_management": "discovery",
    "replace_or_joins": True,
    "prioritize_parallelism": True
}

structure_config_sm3 = {
    "mining_algorithm": "sm3",
    "max_eval_s": 2,
    "concurrency": 0.5,
    "epsilon": 0.15,
    "eta": 0.87,
    "gate_management": "discovery",
    "replace_or_joins": True,
    "prioritize_parallelism": True
}

structure_optimizer_test_data = [
    {
        'name': 'Split Miner 2',
        'config_data': structure_config_sm2
    },
    {
        'name': 'Split Miner 3',
        'config_data': structure_config_sm3
    },
]


@pytest.mark.integration
@pytest.mark.parametrize('test_data', structure_optimizer_test_data,
                         ids=[test_data['name'] for test_data in structure_optimizer_test_data])
def test_miner(entry_point, test_data):
    """Smoke test to check that the structure optimizer can be instantiated and run successfully."""
    log_path = entry_point / 'PurchasingExample.xes'

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / 'model.bpmn'
        params = HyperoptIterationParams(
            output_dir=Path(tmp_dir),
            provided_model_path=output_path,
            project_name="PurchasingExample",
            optimization_metric=Metric.N_GRAM_DISTANCE,
            gateway_probabilities_method=GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
            mining_algorithm=ProcessModelDiscoveryAlgorithm.from_str(test_data['config_data']['mining_algorithm']),
            concurrency=test_data['config_data']['concurrency'],
            epsilon=test_data['config_data']['epsilon'],
            eta=test_data['config_data']['eta'],
            replace_or_joins=test_data['config_data']['replace_or_joins'],
            prioritize_parallelism=test_data['config_data']['prioritize_parallelism']
        )
        discover_process_model(
            log_path,
            output_path,
            params
        )

        assert output_path.exists()
