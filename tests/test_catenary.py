
import unittest
from .function_tests import FunctionTestCase
from ATL import *

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestCatenary(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    num       = Type(float)
    N         = Size('N')
    x         = Var('x')
    link_w    = Var('link_w')
    K_spring  = Var('K_spring')
    gravity   = Var('gravity')
    i         = IVar('i')

    def spring(a,b):
      disp    = Var('disp')
      dist    = Var('dist')
      k       = IVar('k')
      EPS     = 1e-7
      return Let[
        disp,   Gen[k:2]( a[k] - b[k] ),
        dist,   ATLmath.max( EPS, ATLmath.sqrt(Sum[k:2]( disp[k]*disp[k] )) )
      ]( K_spring * (dist - link_w) )

    E_spring  = Var('E_spring')
    E_gravity = Var('E_gravity')
    end_0     = Expr([0,0])
    end_N     = Expr([1,0])

    E_catenary = Fun('E_catenary',num)[
      # number of nodes
      N,
      # parameters to energy model
      link_w : num, K_spring : num, gravity : num,
      # positions of nodes
      x : num[N,2],
    ]( Let[
                    # first energies for the two endpoints
      E_spring,   ( (N>0)*( spring(end_0,x[0]) + spring(end_N,x[N-1]) ) +
                    # energies for all the points in the middle
                    Sum[i:N]( (i+1<N)*spring(x[i],x[i+1]) )),
      E_gravity,  ( Sum[i:N]( gravity*x[i,1] )),
    ]( E_spring + E_gravity ))

    return E_catenary

  def gen_deriv_sig(self):
    return { 'x' : True }

  def gen_deriv(self):
    num       = Type(float)
    N         = Size('N')
    x, dx     = Var('x'), Var('dx')
    link_w    = Var('link_w')
    K_spring  = Var('K_spring')
    gravity   = Var('gravity')
    i         = IVar('i')

    def Dspring(a,b,da,db):
      disp    = Var('disp')
      dist    = Var('dist')
      clamped = Var('clamped')
      d_disp  = Var('d_disp')
      d_dist  = Var('d_dist')
      d_clamped = Var('d_clamped')
      k       = IVar('k')
      EPS     = 1e-7
      return Let[
        disp,     Gen[k:2]( a[k] - b[k] ),
        d_disp,   Gen[k:2]( da[k] - db[k] ),
        dist,     ATLmath.sqrt(Sum[k:2]( disp[k]*disp[k] )),
        d_dist,   Sum[k:2]( disp[k]*d_disp[k] ) / dist,
        clamped,  ATLmath.max( EPS, dist ),
        d_clamped,ATLmath.select_gt( EPS, dist, 0, d_dist ),
      ]( ( K_spring * (clamped - link_w),
           K_spring * d_clamped ) )

    E_spring    = Var('E_spring')
    E_gravity   = Var('E_gravity')
    dE_spring   = Var('dE_spring')
    dE_gravity  = Var('dE_gravity')
    zero2       = Expr([0,0])
    end_0       = zero2
    end_N       = Expr([1,0])

    Dspring_0   = Var('Dspring_0')
    Dspring_N   = Var('Dspring_N')
    Dspring_i   = Var('Dspring_i')

    E_catenary = Fun('E_catenary',(num,num))[
      # number of nodes
      N,
      # parameters to energy model
      link_w : num, K_spring : num, gravity : num,
      # positions of nodes
      x : num[N,2], dx : num[N,2],
    ]( Let[
                    # first energies for the two endpoints
      Dspring_0,  (N>0)* Dspring(end_0,x[0],   zero2,dx[0]  ),
      Dspring_N,  (N>0)* Dspring(end_N,x[N-1], zero2,dx[N-1]),
      Dspring_i,  Gen[i:N]( (i+1<N) * Dspring(x[i],x[i+1],dx[i],dx[i+1]) ),
      E_spring,   ( Dspring_0.proj(0) + Dspring_N.proj(0) ) +
                    # energies for all the points in the middle
                    Sum[i:N]( (i+1<N)* Dspring_i[i].proj(0) ),
      dE_spring,  ( Dspring_0.proj(1) + Dspring_N.proj(1) ) +
                    # energies for all the points in the middle
                    Sum[i:N]( (i+1<N)* Dspring_i[i].proj(1) ),
      E_gravity,  ( Sum[i:N]( gravity*x[i,1] )),
      dE_gravity, ( Sum[i:N]( gravity*dx[i,1] )),
    ]( ( E_spring  + E_gravity,
         dE_spring + dE_gravity ) ))

    return E_catenary

  def rand_input(self):
    # this test puts the points of the chain in a line
    # from (0,0) to (1,0), but with jitter introduced
    N         = self.rand.randint(10,20)

    link_w    = 2/(N+2)
    K_spring  = self.rand.uniform(0.8,1.2)
    gravity   = self.rand.uniform(0.5,2.0)

    x         = self.rand.rand_ndarray([N,2])
    for i in range(0,N):
      scale   = (link_w/4)
      x[i,0]  = scale*x[i,0] + (i+1)/(N+2)
      x[i,1]  = scale*x[i,1]

    return (N,  link_w, K_spring, gravity,  x)

  def rand_deriv_input(self):
    N, link_w, K_spring, gravity, x   = self.rand_input()

    scale     = (link_w/2)
    dx        = self.rand.rand_ndarray(
                    [N,2],
                    gen = (lambda self: self.uniform(-scale,scale))
                )

    return (N,  link_w, K_spring, gravity,  x,  dx)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()

