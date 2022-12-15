import pandas as pd
import pytest
from lxml import etree

from extraneous_activity_delays.config import Configuration, SimulationEngine, SimulationModel
from extraneous_activity_delays.enhance_with_delays import HyperOptEnhancer
from simod.configuration import CalendarSettings, GatewayProbabilitiesDiscoveryMethod, CalendarType
from simod.discovery.extraneous_delay_timers import discover_extraneous_delay_timers
from simod.event_log.column_mapping import EventLogIDs, STANDARD_COLUMNS
from simod.simulation.parameters.miner import mine_parameters

test_cases = [
    {
        'name': 'A',
        'log_name': 'LoanApp_sequential_9-5_diffres_filtered.csv',
        'log_ids': STANDARD_COLUMNS,
        'model_name': 'LoanApp_sequential_9-5_diffres_filtered.bpmn',
        'should_have_delays': False,
    },
    {
        'name': 'B',
        'log_name': 'LoanApp_sequential_9-5_timers.csv',
        'log_ids': EventLogIDs(
            start_time='start_time',
            end_time='end_time',
            activity='Activity',
            resource='Resource',
            case='case_id',
        ),
        'model_name': 'LoanApp_sequential_9-5_timers.bpmn',
        'should_have_delays': True,
    },
]


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_extraneous_activity_delays(test_data, entry_point):
    log_path = entry_point / test_data['log_name']
    model_path = entry_point / test_data['model_name']

    log_ids = test_data['log_ids']

    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")

    # removing extra spaces in activity names from the log
    event_log[log_ids.activity] = event_log[log_ids.activity].str.strip()

    # removing extra spaces in activity names from the model
    tree = etree.parse(str(model_path))
    for element in tree.xpath('//bpmn:task', namespaces={'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}):
        element.attrib['name'] = element.attrib['name'].strip()
    tree.write(str(model_path), pretty_print=True)

    case_arrival_settings = CalendarSettings.default()
    resource_settings = CalendarSettings.default()
    resource_settings.discovery_type = CalendarType.DIFFERENTIATED_BY_RESOURCE

    parameters = mine_parameters(
        case_arrival_settings, resource_settings, event_log, log_ids, model_path,
        GatewayProbabilitiesDiscoveryMethod.DISCOVERY)

    parser = etree.XMLParser(remove_blank_text=True)
    bpmn_model = etree.parse(model_path, parser)

    simulation_model = SimulationModel(bpmn_model, parameters.to_dict())

    configuration = Configuration(
        log_ids=log_ids,
        process_name=model_path.stem,
        max_alpha=50.0,
        num_iterations=1,
        simulation_engine=SimulationEngine.PROSIMOS,
    )

    enhancer = HyperOptEnhancer(event_log, simulation_model, configuration)
    enhanced_simulation_model = enhancer.enhance_simulation_model_with_delays()

    enhanced_simulation_model.bpmn_document.write(model_path.with_stem(model_path.stem + '_timers'), pretty_print=True)

    if test_data['should_have_delays']:
        assert 'event_distribution' in enhanced_simulation_model.simulation_parameters
    else:
        assert 'event_distribution' not in enhanced_simulation_model.simulation_parameters


@pytest.mark.parametrize('test_data', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_discover_extraneous_delay_timers(test_data, entry_point):
    log_path = entry_point / test_data['log_name']
    model_path = entry_point / test_data['model_name']

    log_ids = test_data['log_ids']

    event_log = pd.read_csv(log_path)
    event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True)
    event_log[log_ids.resource].fillna("NOT_SET", inplace=True)
    event_log[log_ids.resource] = event_log[log_ids.resource].astype("string")

    # removing extra spaces in activity names from the log
    event_log[log_ids.activity] = event_log[log_ids.activity].str.strip()

    # removing extra spaces in activity names from the model
    tree = etree.parse(str(model_path))
    for element in tree.xpath('//bpmn:task', namespaces={'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}):
        element.attrib['name'] = element.attrib['name'].strip()
    tree.write(str(model_path), pretty_print=True)

    case_arrival_settings = CalendarSettings.default()
    resource_settings = CalendarSettings.default()
    resource_settings.discovery_type = CalendarType.DIFFERENTIATED_BY_RESOURCE

    parameters = mine_parameters(
        case_arrival_settings, resource_settings, event_log, log_ids, model_path,
        GatewayProbabilitiesDiscoveryMethod.DISCOVERY)

    (out_model_path, out_parameters_path) = discover_extraneous_delay_timers(event_log, log_ids, model_path, parameters,
                                                                             num_iterations=1)

    assert out_model_path.exists()
    assert out_parameters_path.exists()
