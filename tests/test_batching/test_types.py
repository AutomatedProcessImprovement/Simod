from simod.batching.types import BatchingRule


def test_serialization_deserialization():
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
                        {"attribute": "batch_size", "condition": ">", "value": "3"},
                        {"attribute": "batch_size", "condition": "<", "value": "5"},
                    ],
                    [
                        {"attribute": "batch_size", "condition": ">", "value": "10"},
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
                "rules": [[{"attribute": "batch_size", "condition": ">", "value": "3"}]],
            },
        },
    ]

    rules = [BatchingRule.from_dict(rule) for rule in batching_discovery_result]

    assert len(rules) == len(batching_discovery_result)
    for i in range(len(rules)):
        assert rules[i].to_dict() == batching_discovery_result[i]
