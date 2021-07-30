import os
import unittest

from click.testing import CliRunner
from simod.cli import main


class TestCLICommands(unittest.TestCase):
    # NOTE: these are mostly general overall long-running tests to check if everything finishes without exceptions

    def test_discoverer_without_model(self):
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['discover', '--log_path', log_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_discoverer_with_model(self):
        model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['discover', '--log_path', log_path, '--model_path', model_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_optimizer_without_model(self):
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        runner = CliRunner()
        try:
            # optimize --log_path inputs/PurchasingExample.xes
            result = runner.invoke(main, ['optimize', '--log_path', log_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_optimizer_new_without_model(self):
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        runner = CliRunner()
        try:
            # optimize --log_path inputs/PurchasingExample.xes --new_replayer
            result = runner.invoke(main, ['optimize', '--log_path', log_path, '--new_replayer'])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_optimizer_new_with_model(self):
        model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        runner = CliRunner()
        try:
            # optimize --log_path inputs/PurchasingExample.xes --model_path inputs/PurchasingExample.bpmn --new_replayer
            result = runner.invoke(main, [
                'optimize', '--log_path', log_path, '--model_path', model_path, '--new_replayer'
            ])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)


if __name__ == '__main__':
    unittest.main()
