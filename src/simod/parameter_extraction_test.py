import unittest

from simod.parameter_extraction import Pipeline


class TestPipeline(unittest.TestCase):
    class Multiplier:
        input: dict
        output: dict

        def __init__(self, input=None, output=None):
            self.input = input
            self.output = output
            self._execute()

        def _execute(self):
            self.output['product'] = self.input['a'] * self.input['b']

    class Adder:
        input: dict
        output: dict

        def __init__(self, input=None, output=None):
            self.input = input
            self.output = output
            self._execute()

        def _execute(self):
            self.output['sum'] = self.input['a'] + self.input['b']

    def test_pipeline(self):
        input = {'a': 2, 'b': 3}
        output = {'product': None}
        pipeline = Pipeline(input=input, output=output)
        pipeline.set_pipeline([self.Multiplier, self.Adder])
        pipeline.execute()

        self.assertEqual(output['product'], 6)
        self.assertEqual(output['sum'], 5)
        self.assertEqual(pipeline.input, input)
        self.assertEqual(pipeline.output, output)

    def test_pipeline_pipeline_failure(self):
        input = {'a': 2, 'b': 3}
        output = {'product': None}
        pipeline = Pipeline(input=input, output=output)

        with self.assertRaises(ValueError):
            pipeline.execute()


if __name__ == '__main__':
    unittest.main()
