from dataclasses import dataclass


@dataclass
class PrioritizationFiringRule:
    attribute: str
    comparison: str
    value: list[str]

    @staticmethod
    def from_prosimos(rule: dict) -> "PrioritizationFiringRule":
        return PrioritizationFiringRule(
            attribute=rule["attribute"],
            comparison=rule["comparison"],
            value=rule["value"],
        )

    def to_prosimos(self) -> dict:
        return {
            "attribute": self.attribute,
            "comparison": self.comparison,
            "value": self.value,
        }


class AndRules:
    _rules: list[PrioritizationFiringRule]

    def __init__(self, rules: list[PrioritizationFiringRule]):
        self._rules = rules

    @staticmethod
    def from_prosimos(and_rules: list[dict]) -> "AndRules":
        return AndRules(
            rules=list(map(PrioritizationFiringRule.from_prosimos, and_rules)),
        )

    def to_prosimos(self) -> list[dict]:
        return list(map(lambda x: x.to_prosimos(), self._rules))


class OrRules:
    _rules: list[AndRules]

    def __init__(self, rules: list[AndRules]):
        self._rules = rules

    @staticmethod
    def from_prosimos(group: list[list[dict]]) -> "OrRules":
        return OrRules(
            rules=list(map(AndRules.from_prosimos, group)),
        )

    def to_prosimos(self) -> list[dict]:
        return list(map(lambda x: x.to_prosimos(), self._rules))


@dataclass
class PrioritizationRule:
    priority_level: int
    rules: OrRules

    @staticmethod
    def from_prosimos(level: dict) -> "PrioritizationRule":
        return PrioritizationRule(
            priority_level=level["priority_level"],
            rules=OrRules.from_prosimos(level["rules"]),
        )

    def to_prosimos(self) -> dict:
        return {
            "priority_level": self.priority_level,
            "rules": self.rules.to_prosimos(),
        }
