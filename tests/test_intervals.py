import pytest

from simod.simulation.parameters.intervals import prosimos_interval_to_interval_safe

test_cases = [
    {
        'name': 'A',
        'interval': {
            'from': 'Monday',
            'to': 'Friday',
            'beginTime': '00:00:00.000000',
            'endTime': '18:00:00.000000',
        },
    }
]


@pytest.mark.parametrize('test_case', test_cases, ids=[test_data['name'] for test_data in test_cases])
def test_prosimos_interval_to_interval_safe(test_case):
    result = prosimos_interval_to_interval_safe(test_case['interval'])
    assert len(result) == 5
