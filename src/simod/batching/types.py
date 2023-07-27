from dataclasses import dataclass
from typing import Union


@dataclass
class BatchingFiringRule:
    attribute: str
    comparison: str
    value: str

    def __eq__(self, other: "BatchingFiringRule") -> bool:
        return self.attribute == other.attribute and self.comparison == other.comparison and self.value == other.value

    @staticmethod
    def from_dict(rule: dict) -> "BatchingFiringRule":
        return BatchingFiringRule(
            attribute=rule["attribute"],
            comparison=rule["comparison"],
            value=rule["value"],
        )

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "comparison": self.comparison,
            "value": self.value,
        }

    @staticmethod
    def from_prosimos(rule: dict) -> "BatchingFiringRule":
        return BatchingFiringRule(
            attribute=BatchingFiringRule._attribute_name_from_prosimos(rule["attribute"]),
            comparison=rule["comparison"],
            value=BatchingFiringRule._attribute_value_from_prosimos_if_week_day(rule["attribute"], rule["value"]),
        )

    def to_prosimos(self) -> dict:
        return {
            "attribute": self._attribute_name_to_prosimos(self.attribute),
            "comparison": self.comparison,
            "value": self._attribute_value_to_prosimos_if_week_day(self.value),
        }

    @staticmethod
    def _attribute_name_from_prosimos(attribute: str) -> str:
        if attribute == "size":
            return "batch_size"
        elif attribute == "daily_hour":
            return "daily_hour"
        elif attribute == "ready_wt":
            return "batch_ready_wt"
        elif attribute == "large_wt":
            return "batch_max_wt"
        else:
            raise Exception(f"Unknown batching firing rule attribute {attribute}")

    @staticmethod
    def _attribute_name_to_prosimos(attribute: str) -> str:
        if attribute == "batch_size":
            return "size"
        elif attribute == "daily_hour":
            return "daily_hour"
        elif attribute == "batch_ready_wt":
            return "ready_wt"
        elif attribute == "batch_max_wt":
            return "large_wt"
        else:
            raise Exception(f"Unknown batching firing rule attribute {attribute}")

    def _attribute_value_to_prosimos_if_week_day(self, value: str) -> str:
        if self.attribute == "week_day":
            return self._week_day_from_int_to_str(int(value))
        return value

    @staticmethod
    def _attribute_value_from_prosimos_if_week_day(attribute: str, value: str) -> str:
        if attribute == "week_day":
            return str(BatchingFiringRule._week_day_from_str_to_int(value))
        return value

    @staticmethod
    def _week_day_from_str_to_int(week_day: str) -> int:
        week_day = week_day.capitalize()

        if week_day == "Monday":
            return 0
        elif week_day == "Tuesday":
            return 1
        elif week_day == "Wednesday":
            return 2
        elif week_day == "Thursday":
            return 3
        elif week_day == "Friday":
            return 4
        elif week_day == "Saturday":
            return 5
        elif week_day == "Sunday":
            return 6

    @staticmethod
    def _week_day_from_int_to_str(week_day: int) -> str:
        if week_day == 0:
            return "Monday"
        elif week_day == 1:
            return "Tuesday"
        elif week_day == 2:
            return "Wednesday"
        elif week_day == 3:
            return "Thursday"
        elif week_day == 4:
            return "Friday"
        elif week_day == 5:
            return "Saturday"
        elif week_day == 6:
            return "Sunday"


class AndRules:
    _rules: list[BatchingFiringRule]

    def __init__(self, rules: list[BatchingFiringRule]):
        self._rules = rules

    def __iter__(self):
        return iter(self._rules)

    def __next__(self):
        return next(self._rules)

    def __eq__(self, other: "AndRules"):
        return self._rules == other._rules

    def __getitem__(self, rule_index: int) -> BatchingFiringRule:
        return self._rules[rule_index]

    @staticmethod
    def from_list(and_rules: list[dict]) -> "AndRules":
        return AndRules([BatchingFiringRule.from_dict(rule) for rule in and_rules])

    def to_list(self) -> list[dict]:
        return [rule.to_dict() for rule in self._rules]

    @staticmethod
    def from_prosimos(and_rules: list[dict]) -> "AndRules":
        return AndRules([BatchingFiringRule.from_prosimos(rule) for rule in and_rules])

    def to_prosimos(self) -> list[dict]:
        result = []
        for rule in self._rules:
            if isinstance(rule.value, list) or isinstance(rule.value, tuple):
                # when there is an interval in the batch output, we transform it to two rules,
                # one stating "greater than", and the other "lower than"
                greater_than_rule = BatchingFiringRule(rule.attribute, ">", rule.value[0])
                lower_than_rule = BatchingFiringRule(rule.attribute, "<", rule.value[1])
                result.append(greater_than_rule.to_prosimos())
                result.append(lower_than_rule.to_prosimos())
            else:
                result.append(rule.to_prosimos())
        return result


class OrRules:
    _rules: list[AndRules]

    def __init__(self, rules: list[AndRules]):
        self._rules = rules

    def __iter__(self):
        return iter(self._rules)

    def __next__(self):
        return next(self._rules)

    def __eq__(self, other: "OrRules"):
        return self._rules == other._rules

    def __getitem__(self, rule_index: int) -> AndRules:
        return self._rules[rule_index]

    @staticmethod
    def from_list(or_rules: list[list[dict]]) -> "OrRules":
        return OrRules([AndRules.from_list(and_rules) for and_rules in or_rules])

    def to_list(self) -> list[list[dict]]:
        return [and_rule.to_list() for and_rule in self._rules]

    @staticmethod
    def from_prosimos(or_rules: list[list[dict]]) -> "OrRules":
        return OrRules([AndRules.from_prosimos(and_rules) for and_rules in or_rules])

    def to_prosimos(self) -> list[list[dict]]:
        return [and_rule.to_prosimos() for and_rule in self._rules]


@dataclass
class BatchingFiringRules:
    confidence: float
    support: float
    rules: OrRules

    def __eq__(self, other: "BatchingFiringRules"):
        return self.confidence == other.confidence and self.support == other.support and self.rules == other.rules

    def __getitem__(self, rule_index: int) -> AndRules:
        return self.rules[rule_index]

    @staticmethod
    def from_dict(rules: dict) -> "BatchingFiringRules":
        return BatchingFiringRules(
            confidence=rules["confidence"], support=rules["support"], rules=OrRules.from_list(rules["rules"])
        )

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "support": self.support,
            "rules": self.rules.to_list(),
        }

    @staticmethod
    def from_prosimos(rules: list) -> "BatchingFiringRules":
        return BatchingFiringRules(confidence=-1.0, support=-1.0, rules=OrRules.from_prosimos(rules))

    def to_prosimos(self) -> list:
        return self.rules.to_prosimos()


@dataclass
class BatchingRule:
    """
    Rule that defines the batching behavior of an activity.
    """

    activity: str
    resources: Union[list[str], None]
    type: str
    batch_frequency: Union[float, None]
    size_distribution: dict[str, int]
    duration_distribution: dict[str, float]
    firing_rules: BatchingFiringRules

    def __eq__(self, other: "BatchingRule") -> bool:
        return (
            self.activity == other.activity
            and self.resources == other.resources
            and self.type == other.type
            and self.batch_frequency == other.batch_frequency
            and self.size_distribution == other.size_distribution
            and self.duration_distribution == other.duration_distribution
            and self.firing_rules == other.firing_rules
        )

    @staticmethod
    def from_dict(rule: dict) -> "BatchingRule":
        """
        Deserializes a BatchingRule from a dict coming from
        batch_processing_discovery.batch_characteristics.discover_batch_processing_and_characteristics.
        """
        return BatchingRule(
            activity=rule["activity"],
            resources=rule["resources"],
            type=rule["type"],
            batch_frequency=rule["batch_frequency"],
            size_distribution=rule["size_distribution"],
            duration_distribution=rule["duration_distribution"],
            firing_rules=BatchingFiringRules(
                confidence=rule["firing_rules"]["confidence"],
                support=rule["firing_rules"]["support"],
                rules=OrRules.from_list(rule["firing_rules"]["rules"]),
            ),
        )

    def to_dict(self) -> dict:
        """
        Serializes the BatchingRule to a dict.
        """
        return {
            "activity": self.activity,
            "resources": self.resources,
            "type": self.type,
            "batch_frequency": self.batch_frequency,
            "size_distribution": self.size_distribution,
            "duration_distribution": self.duration_distribution,
            "firing_rules": self.firing_rules.to_dict(),
        }

    @staticmethod
    def from_prosimos(rule: dict, activities_names_by_id: dict[str, str]) -> "BatchingRule":
        """
        Deserializes a BatchingRule from a dict coming from Prosimos.
        """
        return BatchingRule(
            activity=activities_names_by_id[rule["task_id"]],
            resources=None,
            type=rule["type"],
            batch_frequency=None,
            size_distribution=BatchingRule._distribution_from_prosimos(rule["size_distrib"]),
            duration_distribution=BatchingRule._distribution_from_prosimos(rule["duration_distrib"]),
            firing_rules=BatchingFiringRules(
                confidence=-1.0,
                support=-1.0,
                rules=OrRules.from_prosimos(rule["firing_rules"]),
            ),
        )

    @staticmethod
    def _distribution_from_prosimos(distributions: list[dict[str, Union[int, float]]]) -> dict[str, Union[int, float]]:
        result = {}
        for item in distributions:
            result[str(item["key"])] = item["value"]
        return result

    def to_prosimos(self, activities_ids_by_name: dict[str, str]) -> dict:
        """
        Serializes the BatchingRule to a dict that can be used to create a batch in Prosimos.

        :param activities_ids_by_name: a dict mapping activity names to their IDs in the BPMN model.
        """
        return {
            "task_id": activities_ids_by_name[self.activity],
            # NOTE: resources are not used in Prosimos
            # "resources": self.resources,
            "type": self.type,
            # NOTE: batch_frequency is not used in Prosimos
            # "batch_frequency": self.batch_frequency,
            "size_distrib": self._distribution_items_to_prosimos(self.size_distribution),
            "duration_distrib": self._distribution_items_to_prosimos(self.duration_distribution),
            "firing_rules": self.firing_rules.to_prosimos(),
        }

    @staticmethod
    def _distribution_items_to_prosimos(distribution: dict[str, Union[int, float]]) -> list[dict]:
        return [{"key": key, "value": value} for key, value in distribution.items()]
