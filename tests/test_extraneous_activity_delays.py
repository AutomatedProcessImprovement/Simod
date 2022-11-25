import pandas as pd
import pytest
from lxml import etree

from extraneous_activity_delays.bpmn_enhancer import set_number_instances_to_simulate, set_start_datetime_to_simulate
from extraneous_activity_delays.config import Configuration
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from simod.event_log.column_mapping import STANDARD_COLUMNS

test_cases = [
    {
        'name': 'A',
        'log_name': 'Production.csv',
        'model_name': 'Production.bpmn',
    },
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_extraneous_activity_delays_launch(test_data, entry_point):
    log_path = entry_point / test_data['log_name']
    model_path = entry_point / test_data['model_name']

    log_ids = STANDARD_COLUMNS
    configuration = Configuration(
        log_ids=log_ids,
        num_evaluations=1,
        num_evaluation_simulations=1,
    )

    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")

    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_model = etree.parse(model_path, parser)
    set_number_instances_to_simulate(bpmn_model, len(event_log[configuration.log_ids.case].unique()))
    set_start_datetime_to_simulate(bpmn_model, min(event_log[configuration.log_ids.start_time]))

    enhancer = HyperOptEnhancer(event_log, bpmn_model, configuration)
    enhanced_bpmn_model = enhancer.enhance_bpmn_model_with_delays()

    enhanced_bpmn_model.write(model_path.with_stem(model_path.stem + '_timers'), pretty_print=True)
