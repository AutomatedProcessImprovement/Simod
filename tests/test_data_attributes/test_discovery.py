import os
import glob
import pytest
import pandas as pd
from pix_framework.io.event_log import EventLogIDs
from simod.data_attributes.discovery import discover_data_attributes\

LOG_IDS = EventLogIDs(case="case_id",
                      activity="activity",
                      start_time="start_time",
                      end_time="end_time",
                      resource="resource"
                      )

ASSET_DIR = "data_attributes"
GLOBAL_ATTRIBUTE_LOG_PATHS = "global_attribute_*.csv.gz"
CASE_ATTRIBUTE_LOG_PATHS = "case_attribute*.csv.gz"
EVENT_ATTRIBUTE_LOG_PATHS = "event_attribute*.csv.gz"


@pytest.fixture(scope="module")
def global_log_files(entry_point):
    log_pattern = os.path.join(entry_point, ASSET_DIR, GLOBAL_ATTRIBUTE_LOG_PATHS)
    return glob.glob(log_pattern)


@pytest.fixture(scope="module")
def case_log_files(entry_point):
    log_pattern = os.path.join(entry_point, ASSET_DIR, CASE_ATTRIBUTE_LOG_PATHS)
    return glob.glob(log_pattern)


@pytest.fixture(scope="module")
def event_log_files(entry_point):
    log_pattern = os.path.join(entry_point, ASSET_DIR, EVENT_ATTRIBUTE_LOG_PATHS)
    return glob.glob(log_pattern)


def assert_attributes(log, log_ids, expected_case_attrs, expected_event_attrs, expected_global_attrs, runs=5):
    success_count = 0

    for i in range(runs):
        global_attributes, case_attributes, event_attributes = discover_data_attributes(log, log_ids)
        print(f"try {i}")
        try:
            assert len(global_attributes) == expected_global_attrs, \
                f"Expected {expected_global_attrs} global attributes, found {len(global_attributes)}"
            assert len(case_attributes) == expected_case_attrs, \
                f"Expected {expected_case_attrs} case attributes, found {len(case_attributes)}"
            assert len(event_attributes) == expected_event_attrs, \
                f"Expected {expected_event_attrs} event attributes, found {len(event_attributes)}"
            success_count += 1
        except AssertionError as e:
            print(f"Assertion failed: {e}")

    if success_count < runs // 2:
        raise AssertionError("Majority of runs failed")


def test_discover_global_attributes(entry_point, global_log_files):
    for log_path in global_log_files:
        log = pd.read_csv(log_path, compression="gzip")
        assert_attributes(log, LOG_IDS, expected_case_attrs=0, expected_event_attrs=16, expected_global_attrs=1)


def test_discover_case_attributes(entry_point, case_log_files):
    for log_path in case_log_files:
        log = pd.read_csv(log_path, compression="gzip")
        assert_attributes(log, LOG_IDS, expected_case_attrs=5, expected_event_attrs=0, expected_global_attrs=0)


def test_discover_event_attributes(entry_point, event_log_files):
    for log_path in event_log_files:
        log = pd.read_csv(log_path, compression="gzip")
        assert_attributes(log, LOG_IDS, expected_case_attrs=0, expected_event_attrs=1, expected_global_attrs=0)

