from dataclasses import dataclass
from enum import Enum
from typing import Union


class CaseAttributeType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class GlobalAttributeType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class EventAttributeType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    EXPRESSION = "expression"
    DTREE = "dtree"


@dataclass
class CaseAttribute:
    name: str
    type: CaseAttributeType
    values: Union[list[dict], dict[str, float]]

    @staticmethod
    def from_dict(case_attribute: dict) -> "CaseAttribute":
        """
        Creates a CaseAttribute object from a dictionary returned by data_attribute_discovery.discovery.
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
                "values": self.values
            }


@dataclass
class GlobalAttribute:
    name: str
    type: GlobalAttributeType
    values: Union[list[dict], dict[str, float]]

    @staticmethod
    def from_dict(global_attribute: dict) -> "GlobalAttribute":
        """
        Creates a GlobalAttribute object from a dictionary returned by data_attribute_discovery.discovery.
        """
        return GlobalAttribute(
            name=global_attribute["name"],
            type=GlobalAttributeType(global_attribute["type"]),
            values=global_attribute["values"],
        )

    def to_prosimos(self) -> dict:
        if self.type == GlobalAttributeType.CONTINUOUS:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values,
            }
        else:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values
            }


@dataclass
class EventAttributeDetails:
    name: str
    type: EventAttributeType
    values: Union[list[dict[str, float]], dict[str, Union[str, list[dict[str, float]]]], str]

    @staticmethod
    def from_dict(attribute: dict) -> "EventAttributeDetails":
        """
        Creates an EventAttributeDetails object from a dictionary returned by data_attribute_discovery.discovery.
        """
        return EventAttributeDetails(
            name=attribute["name"],
            type=EventAttributeType(attribute["type"]),
            values=attribute["values"],
        )

    def to_prosimos(self) -> dict:
        if self.type == EventAttributeType.CONTINUOUS:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values,
            }
        elif self.type == EventAttributeType.DISCRETE:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values

            }
        elif self.type == EventAttributeType.EXPRESSION:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values,
            }
        elif self.type == EventAttributeType.DTREE:
            return {
                "name": self.name,
                "type": self.type.value,
                "values": self.values
            }


@dataclass
class EventAttribute:
    event_id: str
    attributes: list[EventAttributeDetails]

    @staticmethod
    def from_dict(event_attribute: dict) -> "EventAttribute":
        """
        Creates an EventAttribute object from a dictionary.
        """
        return EventAttribute(
            event_id=event_attribute["event_id"],
            attributes=[EventAttributeDetails.from_dict(attr) for attr in event_attribute["attributes"]],
        )

    def to_prosimos(self) -> dict:
        return {
            "event_id": self.event_id,
            "attributes": [attr.to_prosimos() for attr in self.attributes],
        }
