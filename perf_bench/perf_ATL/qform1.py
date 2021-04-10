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
    y[i:N]      = Sum[j:N]( A[i,j] * x[j] )
    return Sum[i:N]( x[i] * y[i] )

  G       = qform.grad(x=True).proj(1)
  Hmul    = G.deriv(x=True).proj(1)
  @ATL.func
  def H( N : size, x : num[N], A : num[N,N] ):
    # construct the materialized Hessian by using an indicator function
    H[i:N,j:N]  = Hmul(N,x,A, Gen[k:N]( (i==k)*1 ))[j]
    return H

  H = H.simplify()
  #qform   = qform.simplify()
  return (qform,H)

# --------------------------------------------------------------------------- #

#q,H = gen_funcs()
#print(H.c_str())
#def foo():
#  N = 200
#  x       = np.asfortranarray(np.random.rand(N))
#  A       = np.asfortranarray(np.random.rand(N,N))
#  out     = np.zeros([N,N],order='F')
#  scalar  = np.zeros([1],order='F')
#  q,H = gen_funcs()
#  print(q.hl_cpp_str(N,x,A, output=scalar))

# --------------------------------------------------------------------------- #

def get_time(N,n_iters=default_timing_iters,timeout_sec=10):
  Htimes = None
  qtimes = None

  # inputs
  x       = np.asfortranarray(np.random.rand(N))
  A       = np.asfortranarray(np.random.rand(N,N))
  out     = np.zeros([N,N],order='F')
  scalar  = np.zeros([1],order='F')
  # functions
  qform, H = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  if N * N * n_iters < 1e7:
    n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def qtimes():
    qform(N,x,A, output=scalar)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Htimes():
    H(N,x,A, output=out)

  return (qtimes,Htimes)



if __name__ == "__main__":
  print("qform1 timings as N varies")
  print(f"{'N':8s}  {'qform':8s}       {'H':8s}      Griewank")
  for N in [100,200,400,800,1600,3200,6400]:
    n_iters = 10 # make sure at least 10 iterations run
    basetime, htime   = get_time(N,n_iters=n_iters)
    basetime  = 1e3 * (basetime.avg() or math.inf)
    htime     = 1e3 * (htime.avg() or math.inf)
    print( f"{N:8d}: {basetime:8.3f} ms    {htime:8.3f} ms "
           f"  {htime/basetime:8.3f}" )



