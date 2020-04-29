from __future__ import annotations
import sys
sys.path += '..'
import ATL
from ATL import num

import numpy as np
import time

@ATL.func
def qform( N : size, x : num[N], A : num[N,N] ):
  B[i:N,j:N]  = Sum[k:N]( A[k,i] * A[k,j] )
  y[i:N]      = Sum[j:N]( B[i,j] * x[j] )
  return Sum[i:N]( x[i] * y[i] )

DQ      = qform.deriv(A=True).proj(1).simplify()
#qform   = qform.simplify()

# --------------------------------------------------------------------------- #

print(DQ)
exit(0)

# --------------------------------------------------------------------------- #

# timing loop with random data

def time_for_N(N,niters=10):
  x       = np.asfortranarray(np.random.rand(N))
  A       = np.asfortranarray(np.random.rand(N,N))
  dA      = np.asfortranarray(np.random.rand(N,N))
  scalar  = np.zeros([1],order='F')
  # prime the pump and cause compilation etc.
  DQ(N,x,A,dA, output=scalar)
  qform(N,x,A, output=scalar)

  start   = time.perf_counter()
  for i in range(0,niters):
    DQ(N,x,A,dA, output=scalar)
  stop    = time.perf_counter()
  dt      = stop - start
  dtime   = dt / niters

  start   = time.perf_counter()
  for i in range(0,niters):
    qform(N,x,A, output=scalar)
  stop    = time.perf_counter()
  dt      = stop - start
  basetime= dt / niters

  return 1e3 * basetime, 1e3 * dtime

print("qform2 timings as N varies")
print(f"{'N':8s}  {'qform':8s}       {'DQ':8s}      Griewank")
for N in [100,200,400,800,1600,3200,6400]:
  niters = 10
  if N * N * niters < 1e8:
    niters = int(1e8)//(N*N)
  basetime, dtime   = time_for_N(N,niters=niters)
  print( f"{N:8d}: {basetime:8.3f} ms    {dtime:8.3f} ms "
         f"  {dtime/basetime:8.3f}" )









