from dataclasses import dataclass


@dataclass
class BranchRule:
    attribute: str
    comparison: str
    value: str

    @staticmethod
    def from_dict(data: dict) -> "BranchRule":
        return BranchRule(
            attribute=data["attribute"],
            comparison=data["comparison"],
            value=data["value"]
        )

    def to_dict(self):
        return {
            "attribute": self.attribute,
            "comparison": self.comparison,
            "value": self.value
        }


@dataclass
class BranchRules:
    id: str
    rules: list[list[BranchRule]]

    @staticmethod
    def from_dict(data: dict) -> "BranchRules":
        return BranchRules(
            id=data["id"],
            rules=[
                [BranchRule.from_dict(rule) for rule in rule_set]
                for rule_set in data["rules"]
            ]
        )

    def to_dict(self):
        return {
            "id": self.id,
            "rules": [[rule.to_dict() for rule in rule_set] for rule_set in self.rules]
        }
