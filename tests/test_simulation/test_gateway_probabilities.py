import pytest
from pix_utils.log_ids import EventLogIDs

from simod.event_log.utilities import read
from simod.settings.control_flow_settings import GatewayProbabilitiesMethod
from simod.simulation.parameters.gateway_probabilities import compute_gateway_probabilities


@pytest.mark.parametrize('log_name', ['PurchasingExample.xes'])
def test_compute_gateway_probabilities(entry_point, log_name):
    log_path = entry_point / log_name
    log, log_path_csv = read(log_path)
    log_ids = EventLogIDs(
        case='case:concept:name',
        activity='concept:name',
        resource='org:resource',
        start_time='start_timestamp',
        end_time='time:timestamp'
    )
    model_path = entry_point / 'PurchasingExample.bpmn'

    # Discover with equiprobable
    gateway_probabilities = compute_gateway_probabilities(log, log_ids, model_path, GatewayProbabilitiesMethod.EQUIPROBABLE)
    # Assert equiprobable probabilities
    assert gateway_probabilities is not None
    for gateway in gateway_probabilities:
        total_paths = len(gateway.outgoing_paths)
        for path in gateway.outgoing_paths:
            assert path.probability == 1.0 / total_paths

    # Discover
    gateway_probabilities = compute_gateway_probabilities(log, log_ids, model_path, GatewayProbabilitiesMethod.DISCOVERY)
    # Assert they add up to one
    assert gateway_probabilities is not None
    for gateway in gateway_probabilities:
        total_paths = len(gateway.outgoing_paths)
        sum_probs = 0.0
        for path in gateway.outgoing_paths:
            assert path.probability != 1.0 / total_paths  # Don't have to hold, but it should in Purchasing Example
            sum_probs += path.probability
        assert sum_probs == 1.0

    log_path_csv.unlink()
