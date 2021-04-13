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
  def abar_energy(N: size, M: size, P: size, A: num[N, M], W: num[N, P], H: num[P, M]):
    abar[i:N, j:M] = Sum[k:P](W[i, k] * H[k, j])
    return Sum[i:N, j:M](ln(abar[i, j]) + (A[i, j] / abar[i, j]))

  abar_energy = abar_energy.simplify()
  dEdWH = abar_energy.grad(W=True, H=True).proj(1)

  return (abar_energy, dEdWH)

# --------------------------------------------------------------------------- #

# E,G = gen_funcs()
# print(G.c_str())
# def foo():
#  N = 200
#  M = 200
#  P = 100
#  A       = np.asfortranarray(np.random.rand(N,M))
#  W       = np.asfortranarray(np.random.rand(N,P))
#  H       = np.asfortranarray(np.random.rand(P,M))
#  out     = np.zeros([N,M],order='F')
#  scalar  = np.zeros([1],order='F')
#  E,G = gen_funcs()
#  print(E.hl_cpp_str(N,M,P,A,W,H, output=scalar))

# --------------------------------------------------------------------------- #

def get_time(N,M=None,P=None,n_iters=default_timing_iters,timeout_sec=100):
  if M is None:
    M = N
  if P is None:
    P = N // 2

  # inputs
  A       = np.asfortranarray(np.random.rand(N,M))
  W       = np.asfortranarray(np.random.rand(N,P))
  H       = np.asfortranarray(np.random.rand(P,M))
  ddW     = np.zeros([N,P],order='F')
  ddH     = np.zeros([P,M],order='F')
  scalar  = np.zeros([1],order='F')
  # functions
  E,G = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  if N * M * n_iters < 1e7:
    n_iters = int(1e7)//(N*M)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Etimes():
    E(N,M,P,A,W,H, output=scalar)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Gtimes():
    G(N,M,P,A,W,H, output=(ddW, ddH))

  return (Etimes,Gtimes)


if __name__ == "__main__":
  print("nnmatfac timings as N varies")
  print(f"{'N':8s}  {'abar_energy':8s}       {'energy gradient':8s}      Griewank")
  for N in [100,200,400,800,1600,3200,6400]:
    n_iters = 10 # make sure at least 10 iterations run
    basetime, gtime   = get_time(N,n_iters=n_iters)
    basetime  = 1e3 * (basetime.avg() or math.inf)
    gtime     = 1e3 * (gtime.avg() or math.inf)
    print( f"{N:8d}: {basetime:8.3f} ms    {gtime:8.3f} ms "
           f"  {gtime/basetime:8.3f}" )



