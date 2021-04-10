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

import ATL
from ATL import num

import numpy as np
import math

from global_params import *

def gen_funcs():
  @ATL.func
  def qform( N : size, x : num[N], A : num[N,N] ):
    B[i:N,j:N]  = Sum[k:N]( A[k,i] * A[k,j] )
    y[i:N]      = Sum[j:N]( B[i,j] * x[j] )
    return Sum[i:N]( x[i] * y[i] )

  DQ      = qform.deriv(A=True).proj(1).simplify()
  qform   = qform.simplify()

  return (qform,DQ)

# --------------------------------------------------------------------------- #

print(gen_funcs())
def foo():
  N       = 200
  x       = np.asfortranarray(np.random.rand(N))
  A       = np.asfortranarray(np.random.rand(N,N))
  dA      = np.asfortranarray(np.random.rand(N,N))
  scalar  = np.zeros([1],order='F')
  # functions
  q, DQ = gen_funcs()
  print(q.hl_cpp_str(N,x,A, output=scalar))

# --------------------------------------------------------------------------- #

def get_time(N,n_iters=default_timing_iters,timeout_sec=10):
  Dtimes = None
  qtimes = None

  x       = np.asfortranarray(np.random.rand(N))
  A       = np.asfortranarray(np.random.rand(N,N))
  dA      = np.asfortranarray(np.random.rand(N,N))
  scalar  = np.zeros([1],order='F')
  # functions
  qform, DQ = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  if N * N * n_iters < 1e7:
    n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def qtimes():
    qform(N,x,A, output=scalar)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Dtimes():
    DQ(N,x,A,dA, output=scalar)

  return (qtimes,Dtimes)

if __name__ == "__main__":
  print("qform2 timings as N varies")
  print(f"{'N':8s}  {'qform':8s}       {'DQ':8s}      Griewank")
  for N in [100,200,400,800,1600,3200,6400]:
    n_iters   = 10
    basetime, dtime   = get_time(N,n_iters=n_iters)
    basetime  = 1e3 * (basetime.avg() or math.inf)
    dtime     = 1e3 * (dtime.avg() or math.inf)
    print( f"{N:8d}: {basetime:8.3f} ms    {dtime:8.3f} ms "
           f"  {dtime/basetime:8.3f}" )









