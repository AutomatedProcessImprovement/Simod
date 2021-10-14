import os
import re
import sys
from pathlib import Path

import pytest

from simod.common_routines import remove_outliers, file_contains
from simod.configuration import Configuration
from simod.readers.log_reader import LogReader


@pytest.fixture
def args(entry_point):
    args = [
        {'model_path': Path(os.path.join(entry_point, 'PurchasingExample.bpmn')),
         'log_path': Path(os.path.join(entry_point, 'PurchasingExample.xes'))},
    ]
    return args


def test_remove_outliers(args):
    for arg in args:
        settings = Configuration()
        log_path = arg['log_path']
        log = LogReader(log_path, settings.read_options)
        print(f'Running test for {log_path}')
        result = remove_outliers(log)
        assert result is not None
        assert 'caseid' in result.keys()
        assert 'duration_seconds' not in result.keys()


def test_file_contains(entry_point):
    paths_without_inclusive = [
        Path(os.path.join(entry_point, 'PurchasingExample.bpmn')),
        Path(os.path.join(entry_point, 'Production.bpmn')),
    ]

    paths_with_inclusive = [
        Path(os.path.join(entry_point, 'ProductionTestFileContains.bpmn')),
    ]

    for file_path in paths_without_inclusive:
        assert file_contains(file_path, "exclusiveGateway") is True
        assert file_contains(file_path, "inclusiveGateway") is False

    for file_path in paths_with_inclusive:
        assert file_contains(file_path, "inclusiveGateway") is True


def test_invalid_xml(entry_point):
    _illegal_unichrs = [(0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F),
                        (0x7F, 0x84), (0x86, 0x9F),
                        (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF)]
    # if sys.maxunicode >= 0x10000:  # not narrow build
    _illegal_unichrs.extend([(0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF),
                             (0x3FFFE, 0x3FFFF), (0x4FFFE, 0x4FFFF),
                             (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                             (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF),
                             (0x9FFFE, 0x9FFFF), (0xAFFFE, 0xAFFFF),
                             (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                             (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF),
                             (0xFFFFE, 0xFFFFF), (0x10FFFE, 0x10FFFF)])

    _illegal_ranges = ["%s-%s" % (chr(low), chr(high))
                       for (low, high) in _illegal_unichrs]
    _illegal_xml_chars_RE = re.compile(u'[%s]' % u''.join(_illegal_ranges))

    def find_invalid_xml(file_path: Path):
        if not file_path.exists():
            return None

        with file_path.open('r') as f:
            line = next((line for line in f if _illegal_xml_chars_RE.search(line)), None)

        return line

    paths = [
        Path(os.path.join(entry_point, 'Production.xes')),
    ]

    for path in paths:
        assert find_invalid_xml(path) is None
