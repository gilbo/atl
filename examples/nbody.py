from __future__ import annotations
import sys
sys.path += '..'
import ATL
from ATL import num

from scipy.optimize import minimize
import scipy.sparse.linalg as splinalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


G = 1e-1
# G = 6.67408e-11
SOFTENING_EPS = 1e-1
@ATL.func
def negU_energy(N: size, m: num[N], x: num[N, 3]):
    return 1 * Sum[i:N, j:N] ( (i < j) * G * m[i] * m[j] / sqrt(SOFTENING_EPS + Sum[k:3] ( pow(x[j,k] - x[i,k], 2) )) )


negU_energy = negU_energy.simplify()
force = negU_energy.grad(x=True).proj(1)  # force is negative gradient of potential energy
force = force.simplify()


@ATL.func
def acc(N: size, m: num[N], x: num[N, 3]):
    a[i:N, k:3] = force(N, m, x)[i, k] / m[i]
    return a


@ATL.func
def acc_direct(N: size, m: num[N], x: num[N, 3]):
    alo[i:N, k:3] = Sum[j:N] ( (i < j) * G * m[j] * (x[j, k] - x[i, k]) / sqrt(Sum[l:3] ( pow(x[j,l] - x[i,l], 2) )) )
    ahi[i:N, k:3] = Sum[j:N] ( (i > j) * G * m[j] * (x[j, k] - x[i, k]) / sqrt(Sum[l:3] ( pow(x[j,l] - x[i,l], 2) )) )
    a[i:N, k:3] = alo[i, k] + ahi[i, k]
    return a


acc = acc.simplify()
acc_direct = acc_direct.simplify()
daccdx = acc.deriv(x='dx').proj(1)
daccdx = daccdx.simplify()


@ATL.func
def M_func(N: size, m: num[N], x: num[N, 3], nx: num[N, 3], dt: num):
    m[i:N, k:3] = nx[i, k] - dt * dt * daccdx(N, m, x, nx)[i, k]
    return m


# --------------------------------------------------------------------------- #

print(force)

# --------------------------------------------------------------------------- #

def time_for_N(N, maxt=10, dt=0.1, num_iters=1000, method='forward'):
  m = np.asfortranarray(np.ones(N)) * 20/N
  x0 = np.asfortranarray(np.random.rand(N, 3)) * 6
  x0 -= np.mean(x0)
  v0 = np.asfortranarray(np.random.rand(N, 3))
  v0 -= np.mean(v0 * m[:, np.newaxis], 0) / np.mean(m)
  num_iters = min(num_iters, int(maxt/dt))
  acc_out = np.asfortranarray(np.random.rand(N, 3))

  def do_backwards_euler(x0, v0):
    def to_vec(nx):
      assert nx.shape == (N, 3)
      return nx.flatten(order='F')

    def to_nx(vec):
      assert vec.shape == (N*3,)
      return vec.reshape((N, 3), order='F')

    def linsolve_update(res):
      # print(res)
      pass

    x = x0
    v = v0
    acc(N, m, x, output=acc_out)
    a = acc_out
    sim_out = [x0]
    t = 0
    for iter_idx in range(num_iters):
      print(f'iter {iter_idx}: ')
      nx_forward = x + dt * v
      def matvec(vec):
        nx = to_nx(vec)
        mx = M_func(N, m, x, nx, dt)
        return to_vec(mx)

      mop = splinalg.LinearOperator((N*3, N*3), matvec=matvec)
      res, _ = splinalg.bicgstab(A=mop, b=to_vec(nx_forward), x0=to_vec(x), callback=linsolve_update)

      x = to_nx(res)
      acc(N, m, x, output=acc_out)
      a = acc_out
      v += dt * a
      sim_out.append(x.copy())
      t += dt
    return np.array(sim_out)

  def do_forwards_euler(x0, v0):
    x = x0
    v = v0
    acc(N, m, x, output=acc_out)
    a = acc_out
    sim_out = [x0]
    t = 0
    for iter_idx in range(num_iters):
      print(f'iter {iter_idx}: ')
      v += 0.5 * dt * a
      x += dt * v
      acc(N, m, x, output=acc_out)
      a = acc_out
      v += 0.5 * dt * a
      sim_out.append(x.copy())
      t += dt
    return np.array(sim_out)
  return do_backwards_euler(x0, v0) if method == 'backward' else do_forwards_euler(x0, v0)


def main():
  N = 100
  maxt = 10
  dt = 0.1

  np.random.seed(1)
  # sim_outs = time_for_N(N=N, maxt=maxt, dt=dt, method='forward')
  sim_outs = time_for_N(N=N, maxt=maxt, dt=dt, method='backward')

  fig, ax = plt.subplots()
  ln, = plt.plot([], [], 'ro')

  def init_anim():
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    return ln,

  def update_anim(frame):
    frame_num = int(frame/dt)
    print(f'frame {frame_num}: {frame}')
    xdata = sim_outs[frame_num, :, 0]
    ydata = sim_outs[frame_num, :, 1]
    ln.set_data(xdata, ydata)
    return ln,

  framerate = 10
  ani = FuncAnimation(fig, update_anim, frames=np.linspace(0, maxt, maxt*framerate),
                      init_func=init_anim)
  plt.show()


if __name__ == "__main__":
  main()