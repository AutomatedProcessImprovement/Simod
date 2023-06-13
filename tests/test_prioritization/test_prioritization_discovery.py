from pix_framework.input import read_csv_log
from pix_framework.log_ids import EventLogIDs

from simod.prioritization.discovery import (
    discover_prioritization_rules,
    get_case_attributes,
)
from simod.prioritization.types import PrioritizationRule


def test_prioritization_rules_serialization_deserialization(entry_point):
    rules_dict = {
        "prioritisation_rules": [
            {
                "priority_level": 1,
                "rules": [
                    [
                        {"attribute": "loan_amount", "condition": "in", "value": ["1000", "2000"]},
                        {"attribute": "type", "condition": "=", "value": "BUSINESS"},
                    ],
                    [{"attribute": "loan_amount", "condition": "in", "value": ["2000", "inf"]}],
                ],
            },
            {"priority_level": 2, "rules": [[{"attribute": "loan_amount", "condition": ">", "value": "500"}]]},
        ]
    }

    rules = list(map(PrioritizationRule.from_prosimos, rules_dict["prioritisation_rules"]))
    rules_dict_2 = {"prioritisation_rules": list(map(lambda x: x.to_prosimos(), rules))}

    assert len(rules) == 2
    assert rules_dict == rules_dict_2


def test_get_case_attributes(entry_point):
    log_path = entry_point / "Insurance_claims_train.csv"
    log_ids = EventLogIDs(
        case="case_id", activity="Activity", start_time="start_time", end_time="end_time", resource="Resource"
    )
    log = read_csv_log(log_path, log_ids)

    case_attributes = get_case_attributes(log, log_ids)

    assert len(case_attributes) > 0
    assert "extraneous_delay" in map(lambda x: x["name"], case_attributes)


def test_discover_prioritization_rules(entry_point):
    log_path = entry_point / "Insurance_claims_train.csv"
    log_ids = EventLogIDs(
        case="case_id", activity="activity", start_time="start_time", end_time="end_time", resource="resource"
    )
    log = read_csv_log(log_path, log_ids)

    rules = discover_prioritization_rules(log, log_ids)

    assert len(rules) > 0
