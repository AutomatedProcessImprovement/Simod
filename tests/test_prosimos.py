from bpdfr_simulation_engine.simulation_properties_parser import parse_qbp_simulation_process

import uuid


def test_parse_qbp_simulation_process(entry_point):
    qbp_bpmn_path = entry_point / 'PurchasingExampleSimulationInfo.bpmn'

    id_suffix = uuid.uuid4().__str__()
    stem = f'{qbp_bpmn_path.stem}_{id_suffix}'
    output_path = qbp_bpmn_path.with_stem(stem).with_suffix('.json')

    parse_qbp_simulation_process(qbp_bpmn_path, output_path)
