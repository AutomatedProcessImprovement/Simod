import os
import unittest
from pathlib import Path

from simod.configuration import Configuration
from simod.readers.log_reader import LogReader

from .parameter_extraction_alt import replay_logs, extract_process_graph, execute_simulator_simple


class TestParameterExtractionAlt(unittest.TestCase):
    def test_replay_logs(self):
        model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
        settings.fill_in_derived_fields()

        log = LogReader(log_path, settings.read_options)
        graph = extract_process_graph(model_path)
        traces = log.get_traces()

        try:
            process_stats, conformant_traces = replay_logs(graph, traces, settings)
        except Exception as e:
            self.fail(f'Should not fail, failed with: {e}')

        self.assertTrue(process_stats is not None)
        self.assertTrue(len(conformant_traces) > 0)

    def test_execute_simulator_simple(self):
        bimp_path = os.path.dirname(__file__) + '/../../external_tools/bimp/qbp-simulator-engine.jar'
        model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'
        csv_output_path = os.path.dirname(__file__) + '/../../test_assets/execute_simulator_simple_output.csv'

        # TODO: add instances and start date attributes for QBP
        # <qbp:processSimulationInfo xmlns:qbp="http://www.qbp-simulator.com/Schema201212" id="qbp_f2e10713-29f3-427a-823e-6fbca3ef94f2" processInstances="94" startDateTime="2011-04-05T07:06:59.999999+00:00" currency="EUR">
        try:
            execute_simulator_simple(bimp_path, model_path, csv_output_path)
        except Exception as e:
            self.fail(f'Should not fail, failed with: {e}')

        print("CSV saved to", csv_output_path)
        # clean up
        # os.remove(csv_output_path)


if __name__ == '__main__':
    unittest.main()
