from dataclasses import dataclass
from typing import Union


@dataclass
class FiringRule:
    attribute: str
    condition: str
    value: str

    @staticmethod
    def from_dict(rule: dict) -> "FiringRule":
        return FiringRule(
            attribute=rule["attribute"],
            condition=rule["condition"],
            value=rule["value"],
        )

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "condition": self.condition,
            "value": self.value,
        }

    def to_prosimos(self) -> dict:
        return {
            "attribute": self._attribute_name_to_prosimos(self.attribute),
            "condition": self.condition,
            "value": self._attribute_value_to_prosimos_if_week_day(self.value),
        }

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
            raise attribute

    def _attribute_value_to_prosimos_if_week_day(self, value: str) -> str:
        if self.attribute == "week_day":
            return self._week_day_from_int_to_str(int(value))
        return value

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
    _rules: list[FiringRule]

    def __init__(self, rules: list[FiringRule]):
        self._rules = rules

    def __iter__(self):
        return iter(self._rules)

    def __next__(self):
        return next(self._rules)

    @staticmethod
    def from_list(and_rules: list[dict]) -> "AndRules":
        return AndRules([FiringRule.from_dict(rule) for rule in and_rules])

    def to_list(self) -> list[dict]:
        return [rule.to_dict() for rule in self._rules]

    def to_prosimos(self) -> list[dict]:
        result = []
        for rule in self._rules:
            if isinstance(rule.value, list) or isinstance(rule.value, tuple):
                # when there is an interval in the batch output, we transform it to two rules,
                # one stating "greater than", and the other "lower than"
                greater_than_rule = FiringRule(rule.attribute, ">", rule.value[0])
                lower_than_rule = FiringRule(rule.attribute, "<", rule.value[1])
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

    @staticmethod
    def from_list(or_rules: list[list[dict]]) -> "OrRules":
        return OrRules([AndRules.from_list(and_rules) for and_rules in or_rules])

    def to_list(self) -> list[list[dict]]:
        return [and_rule.to_list() for and_rule in self._rules]

    def to_prosimos(self) -> list[list[dict]]:
        return [and_rule.to_prosimos() for and_rule in self._rules]


@dataclass
class FiringRules:
    confidence: float
    support: float
    rules: OrRules

    @staticmethod
    def from_dict(rules: dict) -> "FiringRules":
        return FiringRules(
            confidence=rules["confidence"], support=rules["support"], rules=OrRules.from_list(rules["rules"])
        )

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "support": self.support,
            "rules": self.rules.to_list(),
        }

    def to_prosimos(self) -> dict:
        return {
            "confidence": self.confidence,
            "support": self.support,
            "rules": self.rules.to_prosimos(),
        }


@dataclass
class BatchingRule:
    """
    Rule that defines the batching behavior of an activity.
    """

    activity: str
    resources: list[str]
    type: str
    batch_frequency: float
    size_distribution: dict[str, int]
    duration_distribution: dict[str, float]
    firing_rules: FiringRules

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
            firing_rules=FiringRules(
                confidence=rule["firing_rules"]["confidence"],
                support=rule["firing_rules"]["support"],
                rules=OrRules(
                    [
                        AndRules(
                            [
                                FiringRule(
                                    rule["attribute"],
                                    rule["condition"],
                                    rule["value"],
                                )
                                for rule in and_rule
                            ]
                        )
                        for and_rule in rule["firing_rules"]["rules"]
                    ]
                ),
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
