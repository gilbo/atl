from __future__ import annotations
import sys
sys.path += '..'
import ATL
from ATL import num

#import math
import numpy as np
import time
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

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

# --------------------------------------------------------------------------- #

print(H)

# --------------------------------------------------------------------------- #

# timing loop with random data

def time_for_N(N,niters=10):
  x       = np.asfortranarray(np.random.rand(N))
  A       = np.asfortranarray(np.random.rand(N,N))
  out     = np.zeros([N,N],order='F')
  scalar  = np.zeros([1],order='F')
  # prime the pump and cause compilation etc.
  H(N,x,A, output=out)
  qform(N,x,A, output=scalar)

  start   = time.perf_counter()
  for i in range(0,niters):
    H(N,x,A, output=out)
  stop    = time.perf_counter()
  dt      = stop - start
  H_time  = dt / niters

  start   = time.perf_counter()
  for i in range(0,niters):
    qform(N,x,A, output=scalar)
  stop    = time.perf_counter()
  dt      = stop - start
  basetime= dt / niters

  return 1e3 * basetime, 1e3 * H_time

print("qform1 timings as N varies")
print(f"{'N':8s}  {'qform':8s}       {'H':8s}      Griewank")
for N in [100,200,400,800,1600,3200,6400]:
  niters = 10
  if N * N * niters < 1e8:
    niters = int(1e8)//(N*N)
  basetime, htime   = time_for_N(N,niters=niters)
  print( f"{N:8d}: {basetime:8.3f} ms    {htime:8.3f} ms "
         f"  {htime/basetime:8.3f}" )



