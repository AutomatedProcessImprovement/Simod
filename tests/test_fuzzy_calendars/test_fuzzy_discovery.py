import pandas as pd
import pytest
from pix_framework.discovery.start_time_estimator.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.discovery.start_time_estimator.config import (
    ConcurrencyThresholds,
)
from pix_framework.discovery.start_time_estimator.config import (
    Configuration as StartTimeEstimatorConfiguration,
)
from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs, read_csv_log

from simod.bpm.graph import get_activities_ids_by_name_from_bpmn
from simod.fuzzy_calendars.discovery import (
    discovery_fuzzy_simulation_parameters,
)


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
    ids=[
        "fuzzy_calendars_log.csv.gz",
        "LoanApp_simplified.csv.gz",
    ],
)
def test_fuzzy_calendar_discovery_from_df(entry_point, test_data):
    """
    Checks if the fuzzy discovery technique doesn't discover any fuzziness in the classic log (LoanApp_simplified.csv),
    and if it discovers fuzziness in the fuzzy log (fuzzy_calendars_log.csv).
    """
    log_path = entry_point / test_data["log_path"]
    log = read_csv_log(log_path, PROSIMOS_LOG_IDS)
    _add_enabled_times(log, PROSIMOS_LOG_IDS)
    bpmn_path = entry_point / test_data["bpmn_path"]
    activities_ids = get_activities_ids_by_name_from_bpmn(bpmn_path)

    # avoiding the applicant with low workload
    log = log[log[PROSIMOS_LOG_IDS.resource] != "Applicant-000001"]

    # discover fuzzy calendars
    result, _ = discovery_fuzzy_simulation_parameters(
        log=log,
        log_ids=PROSIMOS_LOG_IDS,
        task_ids_by_name=activities_ids,
    )

    # calculate error
    numerator = 0
    denominator = 0
    for calendar in result:
        for interval in calendar.intervals:
            if interval.probability < 0.8:
                numerator += 1
            denominator += 1
    error = numerator / denominator

    # check error
    assert error <= test_data["error_threshold"], f"Error: {error} > {test_data['error_threshold']}"


# TODO: integrate fuzzy discovery into Simod as an option.
# TODO: integrate fuzzy discovery into Simod optimization pipeline.
#     There is one missing thing from the FUZZY format that is not
#     here: when outputting to JSON parameters, the FUZZY version
#     generates two extra attributes ('model_type' and 'granule_size'),
#     discuss if this makes sense there, and how to represent it in
#     PIX_FRAMEWORK model.
#     What I would like to do, is keep the BPS model classes from our
#     framework pure, without details that only make sense in Prosimos,
#     and then transform to Prosimos format when needed to output JSON.
#     For example, instead of storing this 'model_type=FUZZY' as part of
#     the BPS model, we store the BPS model as it is, and when we create
#     the JSON parameters, we check that the calendars are fuzzy, so we
#     append this parameter to the output.


def _add_enabled_times(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    # The start times are the original ones, so use overlapping concurrency oracle
    OverlappingConcurrencyOracle(log, configuration).add_enabled_times(log)
