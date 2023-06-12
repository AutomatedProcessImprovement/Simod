from dataclasses import dataclass


@dataclass
class PrioritizationRule:
    attribute: str
    condition: str
    value: list[str]

    @staticmethod
    def from_dict(rule: dict) -> "PrioritizationRule":
        return PrioritizationRule(
            attribute=rule["attribute"],
            condition=rule["condition"],
            value=rule["value"],
        )

    def to_prosimos(self) -> dict:
        return {
            "attribute": self.attribute,
            "condition": self.condition,
            "value": self.value,
        }


@dataclass
class PrioritizationGroup:
    rules: list[PrioritizationRule]

    @staticmethod
    def from_list(group: list[dict]) -> "PrioritizationGroup":
        return PrioritizationGroup(
            rules=list(map(PrioritizationRule.from_dict, group)),
        )

    def to_list(self) -> list[dict]:
        return list(map(lambda x: x.to_prosimos(), self.rules))


@dataclass
class PrioritizationLevel:
    priority_level: int
    rules: list[PrioritizationGroup]

    @staticmethod
    def from_prosimos(level: dict) -> "PrioritizationLevel":
        return PrioritizationLevel(
            priority_level=level["priority_level"],
            rules=list(map(PrioritizationGroup.from_list, level["rules"])),
        )

    def to_prosimos(self) -> dict:
        return {
            "priority_level": self.priority_level,
            "rules": list(map(lambda x: x.to_list(), self.rules)),
        }
