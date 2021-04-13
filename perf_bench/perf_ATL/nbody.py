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
  G = 6.67408e-11
  SOFTENING_EPS = 1e-1
  @ATL.func
  def negU_energy(N: size, m: num[N], x: num[N, 3]):
      return 1 * Sum[i:N, j:N] ( (i < j) * G * m[i] * m[j] / sqrt(pow(SOFTENING_EPS, 2) + Sum[k:3] ( pow(x[j,k] - x[i,k], 2) )) )

  negU_energy = negU_energy.simplify()
  force = negU_energy.grad(x=True).proj(1)  # force is negative gradient of potential energy
  force = force.simplify()

  @ATL.func
  def acc(N: size, m: num[N], x: num[N, 3]):
      a[i:N, k:3] = force(N, m, x)[i, k] / m[i]
      return a


  acc = acc.simplify()
  daccdx = acc.deriv(x='dx').proj(1)
  daccdx = daccdx.simplify()

  return (negU_energy, daccdx)

# --------------------------------------------------------------------------- #

# E,dA = gen_funcs()
# print(M.c_str())
# def foo():
#  N = 200
#  m       = np.asfortranarray(np.random.rand(N))
#  x       = np.asfortranarray(np.random.rand(N,3))
#  out     = np.zeros([N,3],order='F')
#  scalar  = np.zeros([1],order='F')
#  E,dA = gen_funcs()
#  print(E.hl_cpp_str(N,m,x, output=scalar))

# --------------------------------------------------------------------------- #

def get_time(N,n_iters=default_timing_iters,timeout_sec=100):
  # inputs
  m       = np.asfortranarray(np.random.rand(N))
  x       = np.asfortranarray(np.random.rand(N,3))
  nx      = np.asfortranarray(np.random.rand(N,3))
  dx     = np.zeros([N,3],order='F')
  scalar  = np.zeros([1],order='F')
  # functions
  E,dA = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  if N * N * n_iters < 1e7:
    n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Etimes():
    E(N,m,x, output=scalar)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def dAtimes():
    dA(N,m,x,nx, output=(dx))

  return (Etimes,dAtimes)


if __name__ == "__main__":
  print("nbody timings as N varies")
  print(f"{'N':8s}  {'U energy':8s}       {'acceleration derivative':8s}      Griewank")
  for N in [100,200,400,800,1600,3200,6400]:
    n_iters = 10 # make sure at least 10 iterations run
    basetime, gtime   = get_time(N,n_iters=n_iters)
    basetime  = 1e3 * (basetime.avg() or math.inf)
    gtime     = 1e3 * (gtime.avg() or math.inf)
    print( f"{N:8d}: {basetime:8.3f} ms    {gtime:8.3f} ms "
           f"  {gtime/basetime:8.3f}" )



