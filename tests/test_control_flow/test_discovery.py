import tempfile
from pathlib import Path

import pytest
from pix_framework.discovery.gateway_probabilities import GatewayProbabilitiesDiscoveryMethod
from simod.bpm.reader_writer import BPMNReaderWriter
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
    "prioritize_parallelism": True,
}

structure_config_sm3 = {
    "mining_algorithm": "sm3",
    "max_eval_s": 2,
    "concurrency": 0.5,
    "epsilon": 0.15,
    "eta": 0.87,
    "gate_management": "discovery",
    "replace_or_joins": True,
    "prioritize_parallelism": True,
}

structure_config_sm_v1 = {
    "mining_algorithm": "split_miner_v1",
    "max_eval_s": 2,
    "eta": 0.87,
    "epsilon": 0.15,
    "prioritize_parallelism": True,
    "replace_or_joins": True,
    "gate_management": "discovery",
}

structure_config_sm_v2 = {
    "mining_algorithm": "split_miner_v2",
    "max_eval_s": 2,
    "epsilon": 0.5,
    "gate_management": "discovery",
}

structure_optimizer_test_data = [
    {"name": "Split Miner 2", "config_data": structure_config_sm2},
    {"name": "Split Miner 3", "config_data": structure_config_sm3},
]

test_data = [
    {
        "name": "Split Miner 2",
        "config_data": structure_config_sm2,
    },
    {
        "name": "Split Miner 3",
        "config_data": structure_config_sm3,
    },
    {
        "name": "Split Miner v1",
        "config_data": structure_config_sm_v1,
    },
    {
        "name": "Split Miner v2",
        "config_data": structure_config_sm_v2,
    },
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "test_data", structure_optimizer_test_data, ids=[test_data["name"] for test_data in structure_optimizer_test_data]
)
def test_discover_process_model(entry_point, test_data):
    """Smoke test to check that the structure optimizer can be instantiated and run successfully."""
    log_path = entry_point / "PurchasingExample.xes"

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "model.bpmn"
        params = HyperoptIterationParams(
            output_dir=Path(tmp_dir),
            provided_model_path=output_path,
            project_name="PurchasingExample",
            optimization_metric=Metric.TWO_GRAM_DISTANCE,
            gateway_probabilities_method=GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
            mining_algorithm=ProcessModelDiscoveryAlgorithm.from_str(test_data["config_data"]["mining_algorithm"]),
            concurrency=test_data["config_data"]["concurrency"],
            epsilon=test_data["config_data"]["epsilon"],
            eta=test_data["config_data"]["eta"],
            replace_or_joins=test_data["config_data"]["replace_or_joins"],
            prioritize_parallelism=test_data["config_data"]["prioritize_parallelism"],
        )
        discover_process_model(log_path, output_path, params)
        # Assert file exists
        assert output_path.exists()
        # Assert is BPMN readable and has activities
        bpmn_reader = BPMNReaderWriter(output_path)
        activities = bpmn_reader.read_activities()
        assert len(activities) > 0


@pytest.mark.parametrize("test_data", test_data, ids=[test_case["name"] for test_case in test_data])
def test_control_flow_discovery(entry_point, test_data):
    """Smoke test to check that the structure optimizer can be instantiated and run successfully."""
    log_path = entry_point / "PurchasingExample.xes"
    tmp_dir = Path(entry_point) / "control_flow_discovery_output"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    output_path = Path(tmp_dir) / f"model-{test_data['config_data']['mining_algorithm']}.bpmn"
    params = HyperoptIterationParams(
        output_dir=Path(tmp_dir),
        provided_model_path=output_path,
        project_name="PurchasingExample",
        optimization_metric=Metric.TWO_GRAM_DISTANCE,
        gateway_probabilities_method=GatewayProbabilitiesDiscoveryMethod.EQUIPROBABLE,
        mining_algorithm=ProcessModelDiscoveryAlgorithm.from_str(test_data["config_data"]["mining_algorithm"]),
        concurrency=test_data["config_data"]["concurrency"] if "concurrency" in test_data["config_data"] else None,
        epsilon=test_data["config_data"]["epsilon"] if "epsilon" in test_data["config_data"] else None,
        eta=test_data["config_data"]["eta"] if "eta" in test_data["config_data"] else None,
        replace_or_joins=test_data["config_data"]["replace_or_joins"]
        if "replace_or_joins" in test_data["config_data"]
        else None,
        prioritize_parallelism=test_data["config_data"]["prioritize_parallelism"]
        if "prioritize_parallelism" in test_data["config_data"]
        else None,
    )

    discover_process_model(log_path, output_path, params)

    assert output_path.exists()
    bpmn_reader = BPMNReaderWriter(output_path)
    activities = bpmn_reader.read_activities()
    assert len(activities) > 0
