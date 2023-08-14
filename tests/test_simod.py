from pathlib import Path

import pytest
from pix_framework.filesystem.file_manager import get_random_folder_id
from pix_framework.io.event_log import DEFAULT_XES_IDS, PROSIMOS_LOG_IDS, read_csv_log
from simod.event_log.event_log import EventLog
from simod.settings.common_settings import PROJECT_DIR
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
    settings.common.train_log_path = (entry_point / Path(settings.common.train_log_path).name).absolute()

    if settings.common.test_log_path:
        settings.common.test_log_path = (entry_point / Path(settings.common.test_log_path).name).absolute()
    if settings.common.model_path:
        settings.common.model_path = (entry_point / Path(settings.common.model_path).name).absolute()

    event_log = EventLog.from_path(
        train_log_path=settings.common.train_log_path,
        log_ids=settings.common.log_ids,
        test_log_path=settings.common.test_log_path,
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


@pytest.mark.system
def test_missing_activities_repaired(entry_point):
    """
    Tests if missing activities are repaired when a model is provided.
    Issue #129.
    """
    bpmn_path = entry_point / "LoanApp_simplified.bpmn"
    log_path = entry_point / "LoanApp_simplified_without_approve_loan_offer.csv"
    settings = SimodSettings.default()
    settings.common.train_log_path = log_path
    settings.common.model_path = bpmn_path
    settings.common.log_ids = DEFAULT_XES_IDS
    event_log = EventLog.from_path(
        train_log_path=settings.common.train_log_path,
        log_ids=settings.common.log_ids,
        test_log_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
    )
    output_dir = PROJECT_DIR / "outputs" / get_random_folder_id()

    simod = Simod(settings, event_log=event_log, output_dir=output_dir)
    simod.run()

    # Assert not failing
    assert len(simod.final_bps_model.resource_model.activity_resource_distributions) > 0
    # The removed activity (Approve Loan Offer) is part of the simulated logs
    activity_name = "Approve loan offer"
    activity_id = "Activity_1y2vzu0"
    simulated_log_path = output_dir / "best_result/simulation/simulated_log_0.csv"
    df = read_csv_log(simulated_log_path, PROSIMOS_LOG_IDS)
    assert activity_name in df[PROSIMOS_LOG_IDS.activity].unique()
    # The removed activity is part of the simulation parameters
    # (all the resources can perform it,
    resource_model = simod.final_bps_model.resource_model
    for profile in resource_model.resource_profiles:
        for resource in profile.resources:
            assert activity_id in resource.assigned_tasks
    # and there is an entry for it in the activity_resource_distributions)
    activity_distributions = list(
        filter(lambda x: x.activity_id == activity_id, resource_model.activity_resource_distributions)
    )
    assert len(activity_distributions) == 1
