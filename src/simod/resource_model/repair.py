import numpy as np
import pandas as pd
from pix_framework.discovery.resource_calendar_and_performance.resource_activity_performance import (
    ActivityResourceDistribution,
    ResourceDistribution,
)
from pix_framework.discovery.resource_model import ResourceModel
from pix_framework.io.event_log import EventLogIDs
from pix_framework.statistics.distribution import DurationDistribution, get_best_fitting_distribution

from simod.cli_formatter import print_message


def repair_with_missing_activities(
    resource_model: ResourceModel, model_activities: list[str], event_log: pd.DataFrame, log_ids: EventLogIDs
):
    """
    Updates the resource_model with missing activity_resource_distributions for activities that are present in the
    model but not in yet in the resource_model.
    """

    # getting missing activities
    resource_model_activities = [
        distribution.activity_id for distribution in resource_model.activity_resource_distributions
    ]
    missing_activities = [activity for activity in model_activities if activity not in resource_model_activities]

    # add missing activities to each resource's assigned_tasks
    for resource_profile in resource_model.resource_profiles:
        for resource in resource_profile.resources:
            resource.assigned_tasks += missing_activities

    # estimate the duration distribution of the activity from all its occurrences in event_log
    duration_distributions_per_activity = {}
    for activity in missing_activities:
        duration_distributions_per_activity[activity] = estimate_duration_distribution_for_activity(
            activity, event_log, log_ids
        )

    # add the missing activity resource distributions to the resource model for all the resources
    resource_names = [
        resource.id for resource_profile in resource_model.resource_profiles for resource in resource_profile.resources
    ]
    for activity, duration_distribution in duration_distributions_per_activity.items():
        resource_distributions = [
            ResourceDistribution(
                resource_id=resource_name, distribution=duration_distribution.to_prosimos_distribution()
            )
            for resource_name in resource_names
        ]
        resource_model.activity_resource_distributions.append(
            ActivityResourceDistribution(activity_id=activity, activity_resources_distributions=resource_distributions)
        )

    print_message(f"Repaired resource model with missing activities: {missing_activities}")


def estimate_duration_distribution_for_activity(
    activity: str, event_log: pd.DataFrame, log_ids: EventLogIDs
) -> DurationDistribution:
    activity_events = event_log[event_log[log_ids.activity] == activity]
    durations = (activity_events[log_ids.end_time] - activity_events[log_ids.start_time]).values
    durations = [duration for duration in durations if not pd.isna(duration)]
    durations = [duration.astype("timedelta64[s]").astype(np.float64) for duration in durations]

    if len(durations) > 0:
        distribution = get_best_fitting_distribution(durations)
    else:
        distribution = DurationDistribution(name="fix", mean=1)

    return distribution
