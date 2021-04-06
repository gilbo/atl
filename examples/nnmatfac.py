from __future__ import annotations
import sys
sys.path += '..'
import ATL
from ATL import num

from scipy.optimize import minimize
import numpy as np
import time


@ATL.func
def abar(N: size, M: size, P: size, W: num[N, P], H: num[P, M]):
    abar[i:N, j:M] = Sum[k:P] (W[i, k] * H[k, j])
    return abar
@ATL.func
def abar_energy(N: size, M: size, P: size, A: num[N, M], W: num[N, P], H: num[P, M]):
    abar[i:N, j:M] = Sum[k:P] (W[i, k] * H[k, j])
    return Sum[i:N, j:M] ( ln(abar[i, j]) + (A[i, j] / abar[i, j]) )
@ATL.func
def abar_energy_sym(N: size, P: size, A: num[N, N], W: num[N, P]):
    abar[i:N, j:N] = Sum[k:P] (W[i, k] * W[j,k])
    return Sum[i:N, j:N] ( ln(abar[i, j]) + (A[i, j] / abar[i, j]) )


abar = abar.simplify()
abar_energy = abar_energy.simplify()
abar_energy_sym = abar_energy_sym.simplify()
dEdWH = abar_energy.grad(W=True, H=True).proj(1)
dEdWH = dEdWH.simplify()
dEsymdWH = abar_energy_sym.grad(W=True).proj(1)
dEsymdWH = dEsymdWH.simplify()


# --------------------------------------------------------------------------- #

print(abar_energy)

# --------------------------------------------------------------------------- #


def time_for_N(N, M, P, A=None, symmetric=False, niters=10):
  if A is None:
    A       = np.asfortranarray(np.random.rand(N, N if symmetric else M))
  assert A.shape == (N, N if symmetric else M)
  W_init    = np.asfortranarray(np.random.rand(N, P))
  H_init    = np.asfortranarray(np.random.rand(P, M))
  ddW       = np.zeros([N, P], order='F')
  ddH       = np.zeros([P, M], order='F')
  abar_res  = np.zeros([N, M], order='F')
  scalar    = np.zeros([1], order='F')

  def to_vector(w, h):
    assert w.shape == (N, P)
    if symmetric:
      return w.flatten(order='F')
    assert h.shape == (P, M)
    return np.hstack([w.flatten(order='F'), h.flatten(order='F')])

  def to_WH(vec):
    if symmetric:
      assert vec.shape == (N*P,)
      w = vec.reshape((N, P), order='F')
      return w, w.transpose()
    assert vec.shape == (N*P + P*M,)
    return vec[:N*P].reshape((N, P), order='F'), vec[N*P:].reshape((P, M), order='F')

  def do_optimize(w0, h0):
    def loss(x):
      W, H = to_WH(x)
      if symmetric:
        abar_energy_sym(N, P, A, W, output=scalar)
      else:
        abar_energy(N, M, P, A, W, H, output=scalar)
      print(f'loss: {scalar}')
      return scalar

    def jac(x):
      W, H = to_WH(x)
      if symmetric:
        dEsymdWH(N, P, A, W, output=ddW)
      else:
        dEdWH(N, M, P, A, W, H, output=(ddW, ddH))
      return to_vector(ddW, ddH)

    result = minimize(loss, to_vector(w0, h0), jac=jac)
    return to_WH(result.x)

  Wres, Hres = do_optimize(W_init, H_init)
  Hres = np.asfortranarray(Hres)
  print(f'W:\n{Wres}')
  print(f'H:\n{Hres}')
  print(f'A:\n{A}')
  abar(N, M, P, Wres, Hres, output=abar_res)
  print(f'abar:\n{abar_res}')
  abar_energy(N, M, P, A, Wres, Hres, output=scalar)
  print(f'loss: {scalar}')

  return

Atest = np.array([
  [.1, .2, .3],
  [.4, .5, .6],
  [.7, .8, .9]
], order='F')

# np.random.seed(0)
time_for_N(3, 3, 2, A=Atest, symmetric=False)

