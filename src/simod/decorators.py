import functools
import time
import traceback

from hyperopt import STATUS_OK, STATUS_FAIL


def timeit(func=None, rec_name=None):
    """
    Decorator to measure execution times of methods

    Parameters
    ----------
    method : Any method.

    Returns
    -------
    dict : execution time record

    """
    if not func:
        return functools.partial(timeit, rec_name=rec_name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = rec_name if rec_name else kwargs.get('log_name', func.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (func.__name__, (te - ts) * 1000))
        return result

    return wrapper


def safe_exec(method):
    """
    Decorator to safe execute methods and return the state
    ----------
    method : Any method.
    Returns
    -------
    dict : execution status
    """
    def safety_check(*args, **kwargs):
        is_safe = kwargs.get('is_safe')
        if is_safe:
            try:
                method(*args, **kwargs)
            except Exception as e:
                print(e)
                traceback.print_exc()
                is_safe = False
        return is_safe

    return safety_check


def safe_exec_with_values_and_status(method):
    """
    Decorator to safe execute methods and return the state
    ----------
    method : Any method.
    Returns
    -------
    dict : execution status
    """

    def safety_check(*args, **kw):
        status = kw.get('status', method.__name__.upper())
        response = {'values': [], 'status': status}
        if status == STATUS_OK:
            try:
                response['values'] = method(*args, **kw)
            except Exception as e:
                print(e)
                traceback.print_exc()
                response['status'] = STATUS_FAIL
        return response

    return safety_check
