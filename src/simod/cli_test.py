import os
import unittest

from click.testing import CliRunner
from simod.cli import main, discover, optimize


class TestCLICommands(unittest.TestCase):
    def test_discoverer(self):
        model_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.bpmn'
        log_path = os.path.dirname(__file__) + '/../../test_assets/PurchasingExample.xes'

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['discover', '--log_path', log_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        # TODO: fails because of incorrect paths for external_tools


if __name__ == '__main__':
    unittest.main()
