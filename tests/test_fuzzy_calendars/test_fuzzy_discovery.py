import shutil

import pytest

from simod.fuzzy_calendars.fuzzy_discovery import build_fuzzy_calendars

process_files = {
    "loan_SC_LU": {
        "csv_log": "fuzzy/csv_logs/loan_SC_LU.csv",
        "bpmn_model": "fuzzy/bpmn_models/LoanOriginationModel.bpmn",
        "json": "fuzzy/out/json/loan_SC_LU.json",
        "sim_log": "fuzzy/out/prosimos/logs/loan_SC_LU_log.csv",
        "sim_stats": "fuzzy/out/prosimos/stats/loan_SC_LU_stat.csv",
        "start_datetime": "2015-03-06 15:47:26+00:00",
        "total_cases": 1000,
    },
    "loan_SC_HU": {
        "csv_log": "fuzzy/csv_logs/loan_SC_HU.csv",
        "bpmn_model": "fuzzy/bpmn_models/LoanOriginationModel.bpmn",
        "json": "fuzzy/out/json/loan_SC_HU.json",
        "sim_log": "fuzzy/out/prosimos/logs/loan_SC_HU.csv",
        "sim_stats": "fuzzy/out/prosimos/stats/loan_SC_HU.csv",
        "start_datetime": "2015-03-06 15:47:26+00:00",
        "total_cases": 1000,
    },
    "loan_MC_LU": {
        "csv_log": "fuzzy/csv_logs/loan_MC_LU.csv",
        "bpmn_model": "fuzzy/bpmn_models/LoanOriginationModel.bpmn",
        "json": "fuzzy/out/json/loan_MC_LU.json",
        "sim_log": "fuzzy/out/prosimos/logs/loan_MC_LU.csv",
        "sim_stats": "fuzzy/out/prosimos/stats/loan_MC_LU.csv",
        "start_datetime": "2015-03-09 09:00:26+00:00",
        "total_cases": 1000,
    },
    "loan_MC_HU": {
        "csv_log": "fuzzy/csv_logs/loan_MC_HU.csv",
        "bpmn_model": "fuzzy/bpmn_models/LoanOriginationModel.bpmn",
        "json": "fuzzy/out/json/loan_MC_HU.json",
        "sim_log": "fuzzy/out/prosimos/logs/loan_MC_HU.csv",
        "sim_stats": "fuzzy/out/prosimos/stats/loan_MC_HU.csv",
        "start_datetime": "2015-03-06 15:47:26+00:00",
        "total_cases": 1000,
    },
}


def main():
    for model_name in process_files:
        build_fuzzy_calendars(
            process_files[model_name]["csv_log"],
            process_files[model_name]["bpmn_model"],
            process_files[model_name]["json"],
            15,
        )
        break


@pytest.mark.integration
@pytest.mark.parametrize("test_data", process_files, ids=[model_name for model_name in process_files])
def test_build_fuzzy_calendars(entry_point, test_data):
    result = build_fuzzy_calendars(
        entry_point / process_files[test_data]["csv_log"],
        entry_point / process_files[test_data]["bpmn_model"],
        entry_point / process_files[test_data]["json"],
        15,
    )

    assert result is not None
    assert len(result["resource_profiles"]) > 0
    assert len(result["arrival_time_distribution"]) > 0
    assert len(result["arrival_time_calendar"]) > 0
    assert len(result["gateway_branching_probabilities"]) > 0
    assert len(result["task_resource_distribution"]) > 0
    assert len(result["granule_size"]) > 0

    output_dir = entry_point / "fuzzy/out"
    shutil.rmtree(output_dir)
