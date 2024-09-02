import os
import glob
import pprint

import pytest
import pandas as pd
from pathlib import Path
from pix_framework.io.event_log import EventLogIDs
from simod.branch_rules.discovery import discover_branch_rules
from pix_framework.io.bpm_graph import BPMNGraph

LOG_IDS = EventLogIDs(case="case_id",
                      activity="activity",
                      start_time="start_time",
                      end_time="end_time",
                      resource="resource"
                      )

ASSET_DIR = "branch_rules"
XOR_BPMN = "xor.bpmn"
OR_BPMN = "or.bpmn"
XOR_LOG_PATHS = "xor_*.csv.gz"
OR_LOG_PATHS = "or_8.csv.gz"

# total_branch_rules -> How many branches should get rules
# rules_per_branch -> how many single rules should be on that branch (exact number or range)
xor_expected_conditions = {
    "xor_1.csv.gz": {"total_branch_rules": 15, "rules_per_branch": 1},  # Categorical equal probs
    "xor_2.csv.gz": {"total_branch_rules": 3, "rules_per_branch": 1},  # Categorical unbalanced
    "xor_3.csv.gz": {"total_branch_rules": 15, "rules_per_branch": 1},  # Categorical with different probs
    "xor_5.csv.gz": {"total_branch_rules": 15, "rules_per_branch": (1, 3)},  # Numerical intervals
    "xor_6.csv.gz": {"total_branch_rules": 15, "rules_per_branch": (1, 2)},  # Conditions
    "xor_7.csv.gz": {"total_branch_rules": 15, "rules_per_branch": (1, 3)},  # Complex AND and OR conditions
}

or_expected_conditions = {
    "or_1.csv.gz": {"total_branch_rules": 15, "rules_per_branch": 1},  # Categorical equal probs 1 flow only
    "or_2.csv.gz": {"total_branch_rules": 15, "rules_per_branch": (1, 2)},  # Categorical equal probs 2 flow2
    "or_3.csv.gz": {"total_branch_rules": 15, "rules_per_branch": 1},  # Categorical equal probs all flows (warning)
    "or_4.csv.gz": {"total_branch_rules": 3, "rules_per_branch": 1},  # Categorical unbalanced 1 flow only (warning)
    "or_5.csv.gz": {"total_branch_rules": 6, "rules_per_branch": 1},  # Categorical unbalanced 2 flows (warning)
    "or_6.csv.gz": {"total_branch_rules": 15, "rules_per_branch": (1, 3)},  # Categorical unbalanced all flows (warning)
    "or_7.csv.gz": {"total_branch_rules": 15, "rules_per_branch": (1, 2)},  # Numerical with AND operator
    "or_8.csv.gz": {"total_branch_rules": 15, "rules_per_branch": 1},  # Numerical with full range
}


@pytest.fixture(scope="module")
def xor_log_files(entry_point):
    """Fixture to generate full paths for XOR branch condition log files."""
    xor_log_pattern = os.path.join(entry_point, ASSET_DIR, XOR_LOG_PATHS)
    files = glob.glob(xor_log_pattern)
    return [(file, xor_expected_conditions[os.path.basename(file)]) for file in files]


@pytest.fixture(scope="module")
def or_log_files(entry_point):
    or_log_pattern = os.path.join(entry_point, ASSET_DIR, OR_LOG_PATHS)
    files = glob.glob(or_log_pattern)
    return [(file, or_expected_conditions[os.path.basename(file)]) for file in files]


def assert_branch_rules(bpmn_graph, log, log_ids, expected_conditions):
    branch_rules = discover_branch_rules(bpmn_graph, log, log_ids)

    assert len(branch_rules) == expected_conditions["total_branch_rules"], \
        f"Expected {expected_conditions['total_branch_rules']} BranchRules, found {len(branch_rules)}"

    for branch_rule in branch_rules:
        rule_count = len(branch_rule.rules)

        if isinstance(expected_conditions["rules_per_branch"], tuple):
            min_rules, max_rules = expected_conditions["rules_per_branch"]
            assert min_rules <= rule_count <= max_rules, \
                f"Expected between {min_rules} and {max_rules} rules, found {rule_count}"
        else:
            assert rule_count == expected_conditions["rules_per_branch"], \
                f"Expected {expected_conditions['rules_per_branch']} rules, found {rule_count}"


def test_discover_xor_branch_rules(entry_point, xor_log_files):
    bpmn_path = os.path.join(entry_point, ASSET_DIR, XOR_BPMN)
    for log_path, expected_conditions in xor_log_files:
        log = pd.read_csv(log_path, compression="gzip")
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
        assert_branch_rules(bpmn_graph, log, LOG_IDS, expected_conditions)


def test_discover_or_branch_rules(entry_point, or_log_files):
    bpmn_path = os.path.join(entry_point, ASSET_DIR, OR_BPMN)
    for log_path, expected_conditions in or_log_files:
        log = pd.read_csv(log_path, compression="gzip")
        bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))
        assert_branch_rules(bpmn_graph, log, LOG_IDS, expected_conditions)
