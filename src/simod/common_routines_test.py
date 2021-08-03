import os
import unittest

# from .common_routines import execute_simulator_simple


# class TestCommonRoutines(unittest.TestCase):
    # def test_replay_logs(self):
    #     model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'
    #     log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'
    #
    #     settings = Configuration(model_path=Path(model_path), log_path=Path(log_path))
    #     settings.fill_in_derived_fields()
    #
    #     log = LogReader(log_path, settings.read_options)
    #     graph = extract_process_graph(model_path)
    #     traces = log.get_traces()
    #
    #     try:
    #         process_stats, conformant_traces = replay_logs(graph, traces, settings)
    #     except Exception as e:
    #         self.fail(f'Should not fail, failed with: {e}')
    #
    #     self.assertTrue(process_stats is not None)
    #     self.assertTrue(len(conformant_traces) > 0)

    # def test_execute_simulator_simple(self):
    #     bimp_path = os.path.dirname(__file__) + '/../../external_tools/bimp/qbp-simulator-engine.jar'
    #     model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExampleQBPWithStartDate.bpmn'
    #     csv_output_path = os.path.dirname(__file__) + '/../../test_assets/execute_simulator_simple_output.csv'
    #
    #     try:
    #         execute_simulator_simple(bimp_path, model_path, csv_output_path)
    #     except Exception as e:
    #         self.fail(f'Should not fail, failed with: {e}')
    #
    #     print("CSV saved to", csv_output_path)
    #
    #     if os.path.exists(csv_output_path):
    #         os.remove(csv_output_path)


if __name__ == '__main__':
    unittest.main()
