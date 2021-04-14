from __future__ import annotations

import os
import sys
_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR       = os.path.abspath(os.path.join(_HERE_DIR,"..",".."))
_PERF_DIR       = os.path.abspath(os.path.join(_HERE_DIR,".."))
if _BASE_DIR not in sys.path:
  sys.path.append(_BASE_DIR)
if _PERF_DIR not in sys.path:
  sys.path.append(_PERF_DIR)

import torch
from torch import autograd as ag
from torch.autograd import functional as agf
import numpy as np
import math

from global_params import *


def gen_funcs():

  def qform(x, A):
    return torch.dot(x, (A @ x))
  
  def H(x, A):
    def f(x):
      return qform(x,A)

    hval = agf.hessian(f, x)
    return hval
  
  return (qform,H)

gen_funcs()

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

def get_time(N,n_iters=default_timing_iters,timeout_sec=10):
  Htimes = None
  qtimes = None

  # inputs
  x       = torch.rand(N, dtype=torch.float64)
  A       = torch.rand(N, N, dtype=torch.float64)
  out     = torch.zeros(N, N, dtype=torch.float64)
  scalar  = torch.zeros(1, dtype=torch.float64)
  # functions
  qform, H = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  if N * N * n_iters < 1e7:
    n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def qtimes():
    scalar = qform(x,A)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Htimes():
    out = H(x,A)

  return (qtimes,Htimes)



if __name__ == "__main__":
  print("qform1 timings as N varies")
  print(f"{'N':8s}  {'qform':8s}       {'H':8s}      Griewank")
  for N in [100,200,400,800]:#,1600,3200,6400]:
    n_iters = 10 # make sure at least 10 iterations run
    basetime, htime   = get_time(N,n_iters=n_iters)
    basetime  = 1e3 * (basetime.avg() or math.inf)
    htime     = 1e3 * (htime.avg() or math.inf)
    print( f"{N:8d}: {basetime:8.3f} ms    {htime:8.3f} ms "
           f"  {htime/basetime:8.3f}" )


