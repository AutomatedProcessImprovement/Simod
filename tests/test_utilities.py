from simod.utilities import parse_single_value_or_interval


def test_parse_single_value_or_interval(entry_point):
    assert parse_single_value_or_interval(1.0) == 1.0
    assert parse_single_value_or_interval(0.23) == 0.23
    assert parse_single_value_or_interval(0.0) == 0.0
    assert parse_single_value_or_interval([0.0, 1.0]) == (0.0, 1.0)
    assert parse_single_value_or_interval([0.32, 0.78]) == (0.32, 0.78)
