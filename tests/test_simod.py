import json
import os
from pathlib import Path

import pytest
from pix_framework.filesystem.file_manager import get_random_folder_id
from pix_framework.io.bpmn import get_activities_names_from_bpmn
from pix_framework.io.event_log import DEFAULT_XES_IDS

from simod.event_log.event_log import EventLog
from simod.settings.common_settings import PROJECT_DIR
from simod.settings.simod_settings import SimodSettings
from simod.simod import Simod
from simod.simulation.parameters.BPS_model import BATCHING_RULES_KEY, PRIORITIZATION_RULES_KEY

test_cases = [
    {
        "name": "Simod basic",
        "config_file": "configuration_simod_basic.yml",
        "expect_extraneous": False,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
        "perform_final_evaluation": True,
    },
    {
        "name": "Simod extraneous",
        "config_file": "configuration_simod_with_extraneous.yml",
        "expect_extraneous": True,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
        "perform_final_evaluation": False,
    },
    {
        "name": "Simod with model",
        "config_file": "configuration_simod_with_model.yml",
        "expect_extraneous": False,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
        "perform_final_evaluation": True,
    },
    {
        "name": "Simod with model & extraneous",
        "config_file": "configuration_simod_with_model_and_extraneous.yml",
        "expect_extraneous": True,
        "expect_batching_rules": False,
        "expect_prioritization_rules": False,
        "perform_final_evaluation": False,
    },
    {
        "name": "Simod with model & prioritization",
        "config_file": "configuration_simod_with_model_and_prioritization.yml",
        "expect_extraneous": False,
        "expect_batching_rules": False,
        "expect_prioritization_rules": True,
        "perform_final_evaluation": True,
    },
    {
        "name": "Simod with model & batching",
        "config_file": "configuration_simod_with_model_and_batching.yml",
        "expect_extraneous": False,
        "expect_batching_rules": True,
        "expect_prioritization_rules": False,
        "perform_final_evaluation": True,
    },
]


@pytest.mark.system
@pytest.mark.parametrize("test_data", test_cases, ids=[test_data["name"] for test_data in test_cases])
def test_simod(test_data, entry_point):
    settings: SimodSettings = SimodSettings.from_path(entry_point / test_data["config_file"])
    settings.common.train_log_path = (entry_point / Path(settings.common.train_log_path).name).absolute()

    if settings.common.test_log_path:
        settings.common.test_log_path = (entry_point / Path(settings.common.test_log_path).name).absolute()
    if settings.common.process_model_path:
        settings.common.process_model_path = (entry_point / Path(settings.common.process_model_path).name).absolute()

    event_log = EventLog.from_path(
        train_log_path=settings.common.train_log_path,
        log_ids=settings.common.log_ids,
        test_log_path=settings.common.test_log_path,
        preprocessing_settings=settings.preprocessing,
        need_test_partition=settings.common.perform_final_evaluation,
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
        # Check if any of the iterations has batching rules
        batching_found = _search_element_in_resource_model_iterations(optimizer._output_dir, BATCHING_RULES_KEY)
        # Assert at least one iteration discovered batching rules
        assert batching_found
    else:
        assert optimizer.final_bps_model.batching_rules is None
    if test_data["expect_prioritization_rules"]:
        # Check if any of the iterations has prioritization rules
        batching_found = _search_element_in_resource_model_iterations(optimizer._output_dir, PRIORITIZATION_RULES_KEY)
        # Assert at least one iteration discovered batching rules
        assert batching_found
    else:
        assert optimizer.final_bps_model.prioritization_rules is None
    if test_data["perform_final_evaluation"]:
        assert (optimizer._best_result_dir / "evaluation").exists()
        assert len(os.listdir(optimizer._best_result_dir / "evaluation")) > 1
    else:
        assert not (optimizer._best_result_dir / "evaluation").exists()


def _search_element_in_resource_model_iterations(output_dir: Path, item: str) -> bool:
    item_found = False
    for subdir, dirs, files in os.walk(output_dir / "resource_model"):
        for file in os.listdir(subdir):
            if file.endswith(".json"):
                with open(subdir + os.sep + file) as f:
                    parameters = json.load(f)
                    if item in parameters and len(parameters[item]) > 0:
                        item_found = True
    return item_found


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
    settings.common.process_model_path = bpmn_path
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
    # The removed activity (Approve Loan Offer) is part of the simulation model
    activity_name = "Approve loan offer"
    activity_id = "Activity_1y2vzu0"
    model_activities = get_activities_names_from_bpmn(simod.final_bps_model.process_model)
    assert activity_name in model_activities
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
