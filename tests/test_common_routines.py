from simod.utilities import file_contains


def test_file_contains(entry_point):
    paths_without_inclusive = [
        entry_point / 'PurchasingExample.bpmn',
    ]

    paths_with_inclusive = [
        entry_point / 'ProductionTestFileContains.bpmn',
    ]

    for file_path in paths_without_inclusive:
        assert file_contains(file_path, "exclusiveGateway") is True
        assert file_contains(file_path, "inclusiveGateway") is False

    for file_path in paths_with_inclusive:
        assert file_contains(file_path, "inclusiveGateway") is True
