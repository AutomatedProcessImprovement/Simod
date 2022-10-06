import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from simod.cli_formatter import print_notice
from simod.utilities import execute_shell_cmd
from .parameters.activity_resources import ActivityResourceDistribution
from .parameters.calendars import Calendar
from .parameters.distributions import Distribution
from .parameters.gateway_probabilities import GatewayProbabilities
from .parameters.resource_profiles import ResourceProfile

PROSIMOS_COLUMN_MAPPING = {  # TODO: replace with EventLogIDs
    'case_id': 'caseid',
    'activity': 'task',
    'enable_time': 'enabled_timestamp',
    'start_time': 'start_timestamp',
    'end_time': 'end_timestamp',
    'resource': 'user'
}


@dataclass
class SimulationParameters:
    """Simulation parameters required by Prosimos."""
    resource_profiles: List[ResourceProfile]
    resource_calendars: List[Calendar]
    task_resource_distributions: List[ActivityResourceDistribution]
    arrival_distribution: Distribution
    arrival_calendar: Calendar
    gateway_branching_probabilities: List[GatewayProbabilities]

    def to_dict(self) -> dict:
        """Dictionary compatible with Prosimos."""
        return {
            'resource_profiles':
                [resource_profile.to_dict() for resource_profile in self.resource_profiles],
            'resource_calendars':
                [calendar.to_dict() for calendar in self.resource_calendars],
            'task_resource_distribution':
                [activity_resources.to_dict() for activity_resources in self.task_resource_distributions],
            'arrival_time_distribution':
                self.arrival_distribution.to_dict(),
            'arrival_time_calendar':
                self.arrival_calendar.to_array(),
            'gateway_branching_probabilities':
                [gateway_probabilities.to_dict() for gateway_probabilities in self.gateway_branching_probabilities]
        }

    def to_json_file(self, file_path: Path) -> None:
        """JSON compatible with Prosimos."""
        with file_path.open('w') as f:
            json.dump(self.to_dict(), f)


@dataclass
class ProsimosSettings:
    """Prosimos simulator settings."""

    bpmn_path: Path
    parameters_path: Path
    output_log_path: Path
    num_simulation_cases: int


def simulate_with_prosimos(settings: ProsimosSettings):
    print_notice(f'Prosimos simulator has been chosen')
    print_notice(f'Number of simulation cases: {settings.num_simulation_cases}')

    args = [
        'diff_res_bpsim', 'start-simulation',
        '--bpmn_path', settings.bpmn_path.__str__(),
        '--json_path', settings.parameters_path.__str__(),
        '--log_out_path', settings.output_log_path.__str__(),
        '--total_cases', str(settings.num_simulation_cases)
    ]

    execute_shell_cmd(args)
