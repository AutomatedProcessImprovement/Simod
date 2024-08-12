from pix_framework.io.event_log import DEFAULT_XES_IDS, read_csv_log
from simod.data_attributes.discovery import discover_data_attributes
from simod.prioritization.discovery import (
    discover_prioritization_rules,
)
from simod.prioritization.types import PrioritizationRule


def test_prioritization_rules_serialization_deserialization(entry_point):
    rules_dict = {
        "prioritisation_rules": [
            {
                "priority_level": 1,
                "rules": [
                    [
                        {"attribute": "loan_amount", "comparison": "in", "value": ["1000", "2000"]},
                        {"attribute": "type", "comparison": "=", "value": "BUSINESS"},
                    ],
                    [{"attribute": "loan_amount", "comparison": "in", "value": ["2000", "inf"]}],
                ],
            },
            {"priority_level": 2, "rules": [[{"attribute": "loan_amount", "comparison": ">", "value": "500"}]]},
        ]
    }

    rules = list(map(PrioritizationRule.from_prosimos, rules_dict["prioritisation_rules"]))
    rules_dict_2 = {"prioritisation_rules": list(map(lambda x: x.to_prosimos(), rules))}

    assert len(rules) == 2
    assert rules_dict == rules_dict_2


def test_discover_prioritization_rules(entry_point):
    log_path = entry_point / "Simple_log_with_prioritization.csv"
    log_ids = DEFAULT_XES_IDS
    log = read_csv_log(log_path, log_ids)

    global_attributes, case_attributes, event_attributes = discover_data_attributes(log, log_ids)

    rules = discover_prioritization_rules(log, log_ids, case_attributes)

    assert len(rules) > 0
