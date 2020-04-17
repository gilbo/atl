from __future__ import annotations
import sys
sys.path += '..'
import ATL
from ATL import num

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



EPS     = 1e-7
P       = ATL.Type([('link_w',num), ('K_spring',num), ('gravity',num)])

# define spring energy
@ATL.func
def spring( a : num[2], b : num[2], p : P ):
  disp[k:2] = a[k] - b[k]
  dist      = sqrt(max( EPS, Sum[k:2](disp[k]*disp[k]) )) - p.link_w
  return 0.5 * p.K_spring * dist * dist

# define overall catenary suspension potential energy
@ATL.func
def U( N : size, p : P, x : num[N,2] ) -> num:
  E_spring  = ( (N>0)*( spring([0,0],x[0],  p)
                      + spring([1,0],x[N-1],p) )
              + Sum[i:N]( (i+1<N)*spring(x[i],x[i+1],p) ) )
  E_gravity = Sum[i:N]( p.gravity * x[i,1] )
  return E_spring + E_gravity

# the force is the negated gradient of this potential energy;
GU = U.grad(x=True).proj(1)
@ATL.func
def F( N : size, p : P, x : num[N,2] ):
  f[i:N,k:2] = -1.0 * GU(N, p, x)[i,k]
  return f



# --------------------------------------------------------------------------- #
# The catenary curve has an analytic form that we define independently here

def cat_func(a, x_off, y_off, x):
  return (y_off - a) + a * math.cosh( (x - x_off) / a )

# below, we will want to set x_off = 0.5 and find a value of `a` that
# achieves a specific y_off value for the assumption that f(0) = f(1) = 0.
# We can do this via binary search.

def find_cat(y):
  y = float(y)
  if y < 0.0:
    lo,hi = 100.0, 1e-3
  else:
    lo,hi = -1e-3, -100.0

  while lo - hi > 1.0e-8:
    mid   = (hi+lo)/2.0
    y_lo  = cat_func(lo,0.5,y,0.0)
    y_hi  = cat_func(hi,0.5,y,0.0)
    y_mid = cat_func(mid,0.5,y,0.0)
    assert y_hi >= 0.0 and y_lo <= 0.0
    if y_mid < 0.0:
      lo  = mid
    else:
      hi  = mid

  a = (hi+lo)/2.0
  return lambda x: cat_func(a,0.5,y,x)

# testing that...
cat = find_cat(-1.93)
#print([ cat(i/10) for i in range(0,11) ])
#del cat

# parabolic reference
par = lambda x: 7.76 * (x-0.5)**2 -1.93

def plot_ref(ax):
  # reference catenary curve...
  xs  = [ i/100 for i in range(0,101) ]
  ys  = [ cat(x) for x in xs ]
  ax.plot(xs,ys)

  # reference parabolic curve
  xs  = [ i/100 for i in range(0,101) ]
  ys  = [ par(x) for x in xs ]
  ax.plot(xs,ys)


# --------------------------------------------------------------------------- #
# let's look at a simple forward-Euler integrated simulation
# to convince ourselves we're not crazy.

N         = 10
link_w    = 1.0 * (1/(N+1))
K_spring  = 100.0
gravity   = 9.8
timestep  = 1/30
sub_steps = 10
damp      = 0.01
print(f"N         {N}\n"
      f"link_w    {link_w}\n"
      f"K_spring  {K_spring}\n"
      f"gravity   {gravity}\n"
      f"timestep  {timestep}\n"
      f"sub_steps {sub_steps}\n"
      f"damp      {damp}\n")
p         = (link_w,K_spring,gravity)

x         = np.ndarray([N,2],order='F')
v         = np.zeros([N,2],order='F')
for i in range(0,N):
  x[i]    = [(i+1)/(N+1),0]

@ATL.func
def FwdEulerStep( N : size, p : P, x : num[N,2], v : num[N,2] ):
  a                 = F(N,p, x)
  x[i:N,k:2]        = x[i,k] + (timestep/sub_steps) * v[i,k]
  v[i:N,k:2]        = (1.0 - damp) * (v[i,k] + (timestep/sub_steps) * a[i,k])
  return x,v

FwdEulerStep = FwdEulerStep.simplify()

# visualization
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

def animate(step):
  global x
  global v
  for k in range(0,sub_steps):
    x,v = FwdEulerStep(N,p, x,v)

  xs  = [ 0 ] + [ p[0] for p in x ] + [ 1 ]
  ys  = [ 0 ] + [ p[1] for p in x ] + [ 0 ]
  ax.clear()
  ax.plot(xs,ys)

  plot_ref(ax)

#ani = animation.FuncAnimation(fig, animate, frames=12*int(1/timestep),
#                                            interval=1000*timestep)
#plt.show()
plt.close(fig)

# --------------------------------------------------------------------------- #

#     In the above simulation we can now see that gravity certainly
#     invalidates the assumptions of the catenary curve, and also fails
#     to match a parabolic curve

#     Let's save the static state found by forward Euler simulation
#     now for future reference...

# run 12 seconds of real time animation at least
print("finding reference solution...")
for i in range(0,12*int(1/timestep)):
  animate(i)
fwd_euler_ref_solution = x
print("Forward Euler Solution")
print(fwd_euler_ref_solution)

# --------------------------------------------------------------------------- #

# Now let's try to develop a Backwards Euler Integrator

# Recall that given position x and velocity v, we have the following
# system of differential equations
#
#   dv / dt  = M^-1 F(x)      M^-1 is the inverse mass matrix, which is
#                             the identity matrix in our example/model
#   dx / dt  = v
#
# This is discretized (with linearization of force) in backwards Euler as
#
#   v1 - v0 = dt * (F(x0) + D[F](x0;x1-x0))
#   x1 - x0 = dt * v1
#
# Re-arranging this now linear system to solve for the unknown v1 we get
#
#                           v1 = v0 + dt * (F(x0) + D[F](x0; dt*v1))
#   (I - dt^2 * D[F](x0)) @ v1 = v0 + dt*F(x0)
#   
# which is the linear system Ax = b with
#
#   A = I - dt^2 * D[F](x0)
#   b = v0 + dt*F(x0)
#

# Let's build this system using a linear approximation of the force.

DF    = F.deriv(x='dx')

@ATL.func
def BkwdEuler_b( N : size, p : P, x0 : num[N,2], v0 : num[N,2] ):
  dt            = timestep
  b[i:N,k:2]    = v0[i,k] + dt * F(N,p,x0)[i,k]
  return b

@ATL.func
def BkwdEuler_A_x( N : size, p : P, x0 : num[N,2], x : num[N,2] ):
  dt            = timestep
  dt2           = dt * dt
  Ax[i:N,k:2]   = x[i,k] - dt2 * DF(N,p,x0,x)._1[i,k]
  return Ax

# Now, in order to integrate, we'll use a CG solver
class ConjGradSolver:
  def __init__(self, Ax_func, b_func, N, pm, eps_done=1e-6, iter_done=50):
    self._Ax_func   = Ax_func
    self._b_func    = b_func
    self._N         = N
    self._pm        = pm

    self.eps_done   = eps_done
    self.iter_done  = iter_done

    @ATL.func
    def init_solve_0( N : size, pm : P, x0 : num[N,2],
                                        v0 : num[N,2],
                                         x : num[N,2] ):
      r[i:N,k:2]  = b_func(N,pm,x0,v0)[i,k] - Ax_func(N,pm,x0,x)[i,k]
      r_len       = Sum[i:N,k:2]( r[i,k] * r[i,k] )
      p           = r
      return r, r_len, p
    self._init0   = init_solve_0.simplify()

    @ATL.func
    def init_solve_1( N : size, pm : P, x0 : num[N,2],
                                         x : num[N,2],
                                         r : num[N,2],
                                     r_len : num,
                                         p : num[N,2] ):
      # first iteration of loop rotated out
      Ap          = Ax_func(N,pm,x0,p)
      pAp         = Sum[i:N,k:2]( p[i,k] * Ap[i,k] )
      alpha       = r_len / pAp
      x[i:N,k:2]  = x[i,k] + alpha * p[i,k]
      r[i:N,k:2]  = r[i,k] - alpha * Ap[i,k]
      r_len_prev  = r_len
      r_len       = Sum[i:N,k:2]( r[i,k] * r[i,k] )
      beta        = r_len / r_len_prev

      return x, r, r_len, beta
    self._init1   = init_solve_1.simplify()

    @ATL.func
    def loop_body( N : size, pm : P, x0 : num[N,2],
                                      x : num[N,2],
                                      r : num[N,2],
                                  r_len : num,
                                   beta : num,
                                      p : num[N,2] ):
      # rotated loop body
      p[i:N,k:2]  = r[i,k] + beta * p[i,k]
      Ap          = Ax_func(N,pm,x0,p)
      pAp         = Sum[i:N,k:2]( p[i,k] * Ap[i,k] )
      alpha       = r_len / pAp
      x[i:N,k:2]  = x[i,k] + alpha * p[i,k]
      r[i:N,k:2]  = r[i,k] - alpha * Ap[i,k]
      r_len_prev  = r_len
      r_len       = Sum[i:N,k:2]( r[i,k] * r[i,k] )
      beta        = r_len / r_len_prev

      return x, r, r_len, beta, p
    self._loop    = loop_body.simplify()


  def solve(self, x0, v0, x):
    pm                = self._pm # general parameters
    N                 = self._N

    r, r_len, p       = self._init0(N,pm,x0,v0,x)
    if r_len < self.eps_done * self.eps_done:
      print('exit iter 0')
      return x0
    # first loop iteration, rotated
    x,r,r_len,beta    = self._init1(N,pm,x0, x,r,r_len,p)
    if r_len < self.eps_done * self.eps_done:
      print('exit iter 0')
      return x

    for i in range(1,self.iter_done):
      x,r,r_len,beta,p  = self._loop(N,pm,x0, x,r,r_len,beta,p)
      if r_len < self.eps_done * self.eps_done:
        print('exit iter', i)
        return x

    print('exit iter', self.iter_done)
    return x

# Recall from the above discussion that backwards Euler computes the
# velocity update and position updates as...
#
#   A * v1 = b (solve as above)
#   x1 = x0 + dt * v1

@ATL.func
def BkwdEulerPositionUpdate( N : size, x : num[N,2], v : num[N,2] ):
  dt          = timestep
  x[i:N,k:2]  = x[i,k] + dt * v[i,k]
  return x

x         = np.ndarray([N,2],order='F')
v         = np.zeros([N,2],order='F')
for i in range(0,N):
  x[i]    = [(i+1)/(N+1),0]

pcg       = ConjGradSolver(BkwdEuler_A_x, BkwdEuler_b, N,p)


fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

def animate(step):
  global x
  global v
  v       = ((1.0 - damp)**sub_steps) * pcg.solve(x,v,v)
  x       = BkwdEulerPositionUpdate(N,x,v)

  xs  = [ 0 ] + [ p[0] for p in x ] + [ 1 ]
  ys  = [ 0 ] + [ p[1] for p in x ] + [ 0 ]
  ax.clear()
  ax.plot(xs,ys)

  plot_ref(ax)

#animate(0)
#ani = animation.FuncAnimation(fig, animate, interval=1000*timestep)
#plt.show()
plt.close(fig)


# --------------------------------------------------------------------------- #

#     Let's save the static state found by backward Euler simulation
#     now for future reference...

# run 8 seconds of real time animation at least

print("finding Backward Euler reference solution...")
for i in range(0,5*int(1/timestep)):
  animate(i)
bkwd_euler_ref_solution = x
print("Backward Euler Solution")
print(bkwd_euler_ref_solution)

# --------------------------------------------------------------------------- #

#     Lastly, let's use Newton's method to solve the statics problem.
#     This problem is now just an optimization.  Find the position vector
#     x for the system s.t. U(x) is minimized.
#
#     Newton's method iterates the following procedure
#
#       x_next  = x - step * ( Hessian[U](x)^-1 * grad[U](x) )
#
#     Which means solving the linear system
#
#       Hessian[U](x;h) = grad[U](x)
#
#     Then, we can somehow pick a step-size `step` to take.
#     For simplicity here, we'll make it a constant parameter


class NewtonsMethod:
  def __init__(self, U, N, pm,
               eps_done=1e-5, iter_done=50, step_size=1.0):
    self._N         = N
    self._pm        = pm

    self.eps_done   = eps_done
    self.iter_done  = iter_done
    self.step_size  = step_size

    GU            = U.grad(x=True).proj(1) 
    H             = GU.deriv(x='h').proj(1)

    # ignore dx but fit the previous template...
    @ATL.func
    def b_func( N : size, pm : P, x : num[N,2], dx : num[N,2] ):
      y[i:N,k:2]  = GU(N,pm,x)[i,k]
      return y

    self.pcg      = ConjGradSolver(H,b_func,N,pm)

    @ATL.func
    def update(N : size, pm : P, x : num[N,2], h : num[N,2]):
      len2        = step_size * step_size * Sum[i:N,k:2]( h[i,k] * h[i,k] )
      x[i:N,k:2]  = x[i,k] - step_size * h[i,k]
      return x, len2
    self.update   = update.simplify()

  def solve(self, x):
    # variable for holding the step
    h             = np.zeros([N,2],order='F')
    for i in range(0,self.iter_done):
      h           = self.pcg.solve(x,h,h)
      x, len2     = self.update(self._N,self._pm,x,h)
      if len2 < self.eps_done * self.eps_done:
        print("solved Newton Method on step ", i)
        return x


nmethod = NewtonsMethod(U,N,p)

x         = np.ndarray([N,2],order='F')
for i in range(0,N):
  x[i]    = [(i+1)/(N+1),0]

# do the actual solve now
x         = nmethod.solve(x)
newton_ref_solution = x

print("Newton Method Static Solution")
print(x)

def diff(a,b):
  d = ( a[0] - b[0], a[1] - b[1] )
  return math.sqrt(d[0]*d[0] + d[1]*d[1])

print("Forward vs. Newton max-diff")
print(max( diff(x,y) for x,y in zip(fwd_euler_ref_solution,
                                    newton_ref_solution) ))
print("Backward vs. Newton max-diff")
print(max( diff(x,y) for x,y in zip(bkwd_euler_ref_solution,
                                    newton_ref_solution) ))

xs  = [ 0 ] + [ p[0] for p in x ] + [ 1 ]
ys  = [ 0 ] + [ p[1] for p in x ] + [ 0 ]
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)

ax.clear()
ax.plot(xs,ys)

plot_ref(ax)

plt.show(block=fig)
plt.close(fig)









