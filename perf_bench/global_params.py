
#from multiprocessing import Process
import signal
import time
from contextlib import contextmanager



# ------------------------------------------ #
#                   Timing                   #
# ------------------------------------------ #

class TimeStats:
  def __init__(self, **kwargs):
    self._timeout   = True if len(kwargs) == 0 else False
    if not self._timeout:
      self._avg       = kwargs['total'] / kwargs['n_iters']
      self._total     = kwargs['total']
      self._n_iters   = kwargs['n_iters']

  def timeout(self):
    return self._timeout

  def avg(self):
    return self._avg if not self._timeout else None


default_timing_iters = 10

def take_time(n_iters=default_timing_iters):
  """Use as a decorator function, @take_time()"""
  assert type(n_iters) is int

  def timing_decorator(func):
    start     = time.perf_counter()
    for i in range(0,n_iters):
      func()
    stop      = time.perf_counter()
    dt        = stop - start

    return TimeStats(total=dt, n_iters=n_iters)

  return timing_decorator


# ------------------------------------------ #
#               Timeout Code                 #
# ------------------------------------------ #

default_timeout = 5 # seconds

# thanks to https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/
@contextmanager
def timeout(time=default_timeout):
  def raise_timeout(signum, frame):
    raise TimeoutError

  signal.signal(signal.SIGALRM, raise_timeout)
  signal.alarm(time)

  try:
    yield
  except TimeoutError:
    raise TimeoutError
  finally:
    signal.signal(signal.SIGALRM, signal.SIG_IGN)


# ------------------------------------------ #
#               Combo Helper                 #
# ------------------------------------------ #

def take_time_w_timeout(n_iters=default_timing_iters,
                        timeout_sec=default_timeout):
  def decorator(func):
    result = None
    try:
      with timeout(timeout_sec):
        # pre-execute to cause compilation etc.
        func()

        result = take_time(n_iters=n_iters)(func)

    except TimeoutError:
      result = TimeStats()

    return result

  return decorator

