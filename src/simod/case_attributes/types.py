from dataclasses import dataclass
from enum import Enum
from typing import Union


class CaseAttributeType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass
class CaseAttribute:
    name: str
    type: CaseAttributeType
    values: Union[list[dict], dict[str, float]]

    @staticmethod
    def from_dict(case_attribute: dict) -> "CaseAttribute":
        """
        Creates a CaseAttribute object from a dictionary returned by case_attribute_discovery.discovery.
        """
        return CaseAttribute(
            name=case_attribute["name"],
            type=CaseAttributeType(case_attribute["type"]),
            values=case_attribute["values"],
        )

    def to_prosimos(self) -> dict:
        if self.type == CaseAttributeType.CONTINUOUS:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values,
            }
        else:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": [{"key": value["key"], "value": value["probability"]} for value in self.values],
            }
