import shutil

import pytest
from pix_framework.io.event_log import PROSIMOS_LOG_IDS, read_csv_log
from simod.fuzzy_calendars.discovery import (
    build_fuzzy_calendars,
    discovery_fuzzy_simulation_parameters,
    event_list_from_csv,
    event_list_from_df,
)

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


@pytest.mark.smoke
@pytest.mark.parametrize("test_data", process_files, ids=[model_name for model_name in process_files])
def test_fuzzy_calendar_discovery_runs(entry_point, test_data):
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


def test_event_list_from_csv_and_df(entry_point):
    log_path = entry_point / "fuzzy/csv_logs/loan_MC_HU.csv"

    result_1 = event_list_from_csv(log_path)

    log = read_csv_log(log_path, PROSIMOS_LOG_IDS)
    log[PROSIMOS_LOG_IDS.case] = log[PROSIMOS_LOG_IDS.case].astype(str)

    result_2 = event_list_from_df(log, PROSIMOS_LOG_IDS)

    # sort lists
    result_1 = sorted(result_1, key=lambda x: x.p_case)
    for trace in result_1:
        trace.event_list = sorted(trace.event_list, key=lambda x: x.started_at)
    result_2 = sorted(result_2, key=lambda x: x.p_case)
    for trace in result_2:
        trace.event_list = sorted(trace.event_list, key=lambda x: x.started_at)

    # compare lists
    for i in range(len(result_1)):
        assert result_1[i].p_case == result_2[i].p_case
        assert result_1[i].started_at == result_2[i].started_at
        assert result_1[i].completed_at == result_2[i].completed_at
        assert result_1[i].completed_at == result_2[i].completed_at
        for ii in range(len(result_1[i].event_list)):
            assert result_1[i].event_list[ii].__dict__ == result_2[i].event_list[ii].__dict__


@pytest.mark.parametrize(
    "test_data",
    [
        {
            "log_path": "fuzzy/csv_logs/fuzzy_calendars_log.csv.gz",
            "bpmn_path": "fuzzy/bpmn_models/fuzzy_calendars_model.bpmn",
            "error_threshold": 0.95,
        },
        {
            "log_path": "fuzzy/csv_logs/LoanApp_simplified.csv.gz",
            "bpmn_path": "fuzzy/bpmn_models/LoanApp_simplified.bpmn",
            "error_threshold": 0.05,
        },
    ],
    ids=["fuzzy_calendars_log.csv.gz", "LoanApp_simplified.csv.gz"],
)
def test_fuzzy_calendar_discovery(entry_point, test_data):
    """
    Checks if the fuzzy discovery technique doesn't discover any fuzziness in the classic log (LoanApp_simplified.csv),
    and if it discovers fuzziness in the fuzzy log (fuzzy_calendars_log.csv).
    """
    log_path = entry_point / test_data["log_path"]
    bpmn_path = entry_point / test_data["bpmn_path"]
    output_path = entry_point / "fuzzy/out/json" / log_path.with_suffix(".json").name

    # preprocess timestamps for fuzzy calendar discovery
    log = read_csv_log(log_path, PROSIMOS_LOG_IDS)
    log[PROSIMOS_LOG_IDS.start_time] = log[PROSIMOS_LOG_IDS.start_time].dt.tz_convert("UTC")
    log[PROSIMOS_LOG_IDS.end_time] = log[PROSIMOS_LOG_IDS.end_time].dt.tz_convert("UTC")
    log_path_processed = (
        entry_point / "fuzzy/csv_logs" / log_path.with_stem(f"{log_path.stem}_processed").with_suffix(".csv").name
    )
    log = log[log[PROSIMOS_LOG_IDS.resource] != "Applicant-000001"]  # avoiding the applicant with low workload
    log.to_csv(log_path_processed, index=False)

    # discover fuzzy calendars
    result = build_fuzzy_calendars(log_path_processed, bpmn_path, output_path, 15)

    # calculate error
    numerator = 0
    denominator = 0
    for calendar in result["resource_calendars"]:
        for week_day in calendar["availability_probabilities"]:
            for interval in week_day["fuzzy_intervals"]:
                if interval["probability"] < 0.8:
                    numerator += 1
                denominator += 1
    error = numerator / denominator

    # check error
    assert error <= test_data["error_threshold"], f"Error: {error} > {test_data['error_threshold']}"

    # clean up
    log_path_processed.unlink()
    output_dir = entry_point / "fuzzy/out"
    shutil.rmtree(output_dir)


@pytest.mark.parametrize(
    "test_data",
    [
        {
            "log_path": "fuzzy/csv_logs/fuzzy_calendars_log.csv.gz",
            "bpmn_path": "fuzzy/bpmn_models/fuzzy_calendars_model.bpmn",
            "error_threshold": 0.95,
        },
        {
            "log_path": "fuzzy/csv_logs/LoanApp_simplified.csv.gz",
            "bpmn_path": "fuzzy/bpmn_models/LoanApp_simplified.bpmn",
            "error_threshold": 0.05,
        },
    ],
    ids=["fuzzy_calendars_log.csv.gz", "LoanApp_simplified.csv.gz"],
)
def test_fuzzy_calendar_discovery_from_df(entry_point, test_data):
    """
    Checks if the fuzzy discovery technique doesn't discover any fuzziness in the classic log (LoanApp_simplified.csv),
    and if it discovers fuzziness in the fuzzy log (fuzzy_calendars_log.csv).
    """
    log_path = entry_point / test_data["log_path"]
    bpmn_path = entry_point / test_data["bpmn_path"]
    log = read_csv_log(log_path, PROSIMOS_LOG_IDS)
    # avoiding the applicant with low workload
    log = log[log[PROSIMOS_LOG_IDS.resource] != "Applicant-000001"]

    # discover fuzzy calendars
    result, _ = discovery_fuzzy_simulation_parameters(
        log=log,
        log_ids=PROSIMOS_LOG_IDS,
        bpmn_path=bpmn_path,
    )

    # calculate error
    numerator = 0
    denominator = 0
    for calendar in result:
        for week_day in calendar.intervals:
            for in_day_interval in week_day.in_day_intervals:
                if in_day_interval.probability < 0.8:
                    numerator += 1
                denominator += 1
    error = numerator / denominator

    # check error
    assert error <= test_data["error_threshold"], f"Error: {error} > {test_data['error_threshold']}"
