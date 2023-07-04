from simod.batching.types import BatchingRule

batching_discovery_result = [
    {
        "activity": "B",
        "resources": ["Alice"],
        "type": "Sequential",
        "batch_frequency": 0.96,
        "size_distribution": {"3": 48, "1": 2},
        "duration_distribution": {"3": 0.5},
        "firing_rules": {
            "confidence": 1.0,
            "support": 1.0,
            "rules": [
                [
                    {"attribute": "batch_size", "comparison": ">", "value": "3"},
                    {"attribute": "batch_size", "comparison": "<", "value": "5"},
                ],
                [
                    {"attribute": "batch_size", "comparison": ">", "value": "10"},
                ],
            ],
        },
    },
    {
        "activity": "C",
        "resources": ["Bob"],
        "type": "Sequential",
        "batch_frequency": 0.96,
        "size_distribution": {"3": 48, "1": 2},
        "duration_distribution": {"3": 0.5},
        "firing_rules": {
            "confidence": 1.0,
            "support": 1.0,
            "rules": [[{"attribute": "batch_size", "comparison": ">", "value": "3"}]],
        },
    },
]


def test_serialization_deserialization():
    rules = [BatchingRule.from_dict(rule) for rule in batching_discovery_result]

    assert len(rules) == len(batching_discovery_result)
    for i in range(len(rules)):
        assert rules[i].to_dict() == batching_discovery_result[i]


def test_prosimos_serialization():
    rules = [BatchingRule.from_dict(rule) for rule in batching_discovery_result]
    activities_ids_by_name = {"B": "2", "C": "3"}
    activities_names_by_id = {"2": "B", "3": "C"}

    rules_prosimos = [rule.to_prosimos(activities_ids_by_name) for rule in rules]
    rules_from_prosimos = [BatchingRule.from_prosimos(rule, activities_names_by_id) for rule in rules_prosimos]

    # Prosimos doesn't use resources and batch_frequency attributes, so we set them
    # to None to compare. Also the confidence and support of the rules.
    for rule in rules:
        rule.resources = None
        rule.batch_frequency = None
        rule.firing_rules.confidence = -1.0
        rule.firing_rules.support = -1.0

    assert rules == rules_from_prosimos
