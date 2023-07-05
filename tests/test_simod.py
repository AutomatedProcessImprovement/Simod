from pathlib import Path

import pytest

from simod.event_log.event_log import EventLog
from simod.settings.simod_settings import SimodSettings
from simod.simod import Simod

test_cases = [
    {
        "name": "Simod basic",
        "config_file": "configuration_simod_basic.yml",
        "expect_extraneous": False,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
    },
    {
        "name": "Simod extraneous",
        "config_file": "configuration_simod_with_extraneous.yml",
        "expect_extraneous": True,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
    },
    {
        "name": "Simod with model",
        "config_file": "configuration_simod_with_model.yml",
        "expect_extraneous": False,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
    },
    {
        "name": "Simod with model & extraneous",
        "config_file": "configuration_simod_with_model_and_extraneous.yml",
        "expect_extraneous": True,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
    },
    {
        "name": "Simod with model & prioritization",
        "config_file": "configuration_simod_with_model_and_prioritization.yml",
        "expect_extraneous": False,
        "expect_batching_rules": False,
        "expect_prioritization_rules": True,
    },
    {
        "name": "Simod with model & batching",
        "config_file": "configuration_simod_with_model_and_batching.yml",
        "expect_extraneous": False,
        "expect_batching_rules": True,
        "expect_prioritization_rules": False,
    },
]


@pytest.mark.system
@pytest.mark.parametrize("test_data", test_cases, ids=[test_data["name"] for test_data in test_cases])
def test_simod(test_data, entry_point):
    settings: SimodSettings = SimodSettings.from_path(entry_point / test_data["config_file"])
    settings.common.log_path = (entry_point / Path(settings.common.log_path).name).absolute()

    if settings.common.test_log_path:
        settings.common.test_log_path = (entry_point / Path(settings.common.test_log_path).name).absolute()
    if settings.common.model_path:
        settings.common.model_path = (entry_point / Path(settings.common.model_path).name).absolute()

    event_log = EventLog.from_path(
        path=settings.common.log_path,
        log_ids=settings.common.log_ids,
        process_name=settings.common.log_path.stem,
        test_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
    )
    optimizer = Simod(settings, event_log=event_log)
    optimizer.run()

    assert optimizer.final_bps_model.process_model is not None
    assert optimizer.final_bps_model.resource_model is not None
    assert optimizer.final_bps_model.case_arrival_model is not None
    assert optimizer.final_bps_model.case_attributes is not None
    assert len(optimizer.final_bps_model.case_attributes) > 0
    if test_data["expect_extraneous"]:
        assert optimizer.final_bps_model.extraneous_delays is not None
        assert len(optimizer.final_bps_model.extraneous_delays) == 2
    else:
        assert optimizer.final_bps_model.extraneous_delays is None
    if test_data["expect_batching_rules"]:
        assert optimizer.final_bps_model.batching_rules is not None
        assert len(optimizer.final_bps_model.batching_rules) == 1
    else:
        assert optimizer.final_bps_model.batching_rules is None
    if test_data["expect_prioritization_rules"]:
        assert optimizer.final_bps_model.prioritization_rules is not None
        assert len(optimizer.final_bps_model.prioritization_rules) > 0
    else:
        assert optimizer.final_bps_model.prioritization_rules is None
