from operator import itemgetter

from .. import support_utils as sup


def print_stats(log, conformant, traces):
    print('complete traces:', str(len(traces)),
          ', events:', str(len(log.data)), sep=' ')
    print('conformance percentage:',
          str(sup.ffloat((len(conformant) / len(traces)) * 100, 2)) + '%', sep=' ')


def get_traces(data, one_timestamp):
    """
    returns the data splitted by caseid and ordered by start_timestamp
    """
    cases = list(set([x['caseid'] for x in data]))
    traces = list()
    for case in cases:
        order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        trace = sorted(
            list(filter(lambda x: (x['caseid'] == case), data)),
            key=itemgetter(order_key))
        traces.append(trace)
    return traces
