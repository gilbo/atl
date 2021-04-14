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
    return torch.dot(x, (torch.transpose(A,0,1) @ (A @ x)))

  def DQ(x, A, dA):
    def f(A):
      return qform(x,A)
    dq = agf.jvp(f,A,dA)[1]
    return dq

  #def DtQ(x, A, dq):
  #  def f(A):
  #    return qform(x,A)
  #  dA = agf.vjp(f,A,dq)[1]
  #  return dA


  return (qform,DQ)

gen_funcs()

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

def get_time(N,n_iters=default_timing_iters,timeout_sec=10):
  Dtimes = None
  #Dttimes = None
  qtimes = None

  # inputs
  x       = torch.rand(N,     dtype=torch.float64)
  A       = torch.rand(N, N,  dtype=torch.float64)
  dA      = torch.rand(N, N,  dtype=torch.float64)
  #dq      = torch.rand(1,    dtype=torch.float64)[0]
  scalar  = torch.zeros(1,    dtype=torch.float64)
  # functions
  qform, DQ = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  if N * N * n_iters < 1e7:
    n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def qtimes():
    scalar = qform(x,A)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Dtimes():
    scalar = DQ(x,A,dA)

  #@take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  #def Dttimes():
  #  dA = DtQ(x,A,dq)

  return (qtimes,Dtimes)

if __name__ == "__main__":
  print("qform2 timings as N varies")
  print(f"{'N':8s}  {'qform':8s}       {'DQ':8s}      "
        #f" {'DtQ':8s}      "
        f"Griewank (DQ)")
  for N in [100,200,400,800,1600,3200,6400]:
    n_iters   = 10
    basetime, dtime   = get_time(N,n_iters=n_iters)
    basetime  = 1e3 * (basetime.avg() or math.inf)
    dtime     = 1e3 * (dtime.avg() or math.inf)
    #dttime    = 1e3 * (dttime.avg() or math.inf)
    print( f"{N:8d}: {basetime:8.3f} ms    {dtime:8.3f} ms "
           #f"   {dttime:8.3f} ms "
           f"  {dtime/basetime:8.3f}" )


