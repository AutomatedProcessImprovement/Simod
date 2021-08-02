import os
import unittest

from click.testing import CliRunner
from simod.cli import main


class TestCLICommands(unittest.TestCase):
    # NOTE: these are mostly general overall long-running tests to check if everything finishes without exceptions

    entry_point = os.path.dirname(__file__) + '/../../test_assets'

    def test_discoverer_without_model(self):
        config_path = os.path.join(self.entry_point, 'discover_without_model_config.yml')

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['discover', '--config_path', config_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_discoverer_with_model(self):
        config_path = os.path.join(self.entry_point, 'discover_with_model_config.yml')

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['discover', '--config_path', config_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_optimizer_without_model(self):
        config_path = os.path.join(self.entry_point, 'optimize_debug_config.yml')

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['optimize', '--config_path', config_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)  # TODO: this run can fail and it's okay in some cases
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_optimizer_new_without_model(self):
        config_path = os.path.join(self.entry_point, 'optimize_new_without_model_config.yml')

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['optimize', '--config_path', config_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)
        self.assertTrue('Event log could not be imported' not in result.output)

    def test_optimizer_new_with_model(self):
        config_path = os.path.join(self.entry_point, 'optimize_new_with_model_config.yml')

        runner = CliRunner()
        try:
            result = runner.invoke(main, ['optimize', '--config_path', config_path])
        except Exception as e:
            self.fail(f'Failed with {e}')

        print(result.output)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue('Error:' not in result.output)  # TODO: this run can fail and it's okay in some cases
        self.assertTrue('Event log could not be imported' not in result.output)


if __name__ == '__main__':
    unittest.main()
