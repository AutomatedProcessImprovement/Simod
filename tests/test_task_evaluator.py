import pytest

from simod.bpm.reader_writer import BPMNReaderWriter
from simod.configuration import PDFMethod
from simod.discovery.tasks_evaluator import TaskEvaluator
from simod.event_log.column_mapping import EventLogIDs
from simod.event_log.utilities import read


@pytest.mark.parametrize('test_case', [
    {
        'log_name': 'PurchasingExample.xes',
        'model_name': 'PurchasingExample.bpmn',
        'pdf_method': PDFMethod.DEFAULT,
    },
    # 'Production.xes'
])
def test_task_evaluator_undifferentiated_resources(entry_point, test_case):
    log_path = entry_point / test_case['log_name']
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )

    model_path = entry_point / test_case['model_name']

    pdf_method = test_case['pdf_method']

    bpmn_reader = BPMNReaderWriter(model_path)
    process_graph = bpmn_reader.as_graph()

    log['role'] = 'SYSTEM'
    resource_pool_metadata = {
        'id': 'undifferentiated',
        'name': 'undifferentiated'
    }

    evaluator = TaskEvaluator(process_graph, log, log_ids, resource_pool_metadata, pdf_method)

    activities_distributions = evaluator.elements_data

    assert activities_distributions is not None
