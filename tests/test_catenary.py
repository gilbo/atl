from __future__ import annotations

import unittest
from .function_tests import FunctionTestCase
import ATL
from ATL import num

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestCatenary(unittest.TestCase, FunctionTestCase):

  def test_user_level(self):
    self._init_rand()
    E         = self.gen_func_memo()
    DE        = E.deriv( x=True ).proj(1).simplify()
    d2E       = self.gen_deriv()
    I,dI      = self.rand_deriv_input()
    O1        = DE( *(I+dI) )
    O2        = d2E( *(I+dI) )[1]
    np.testing.assert_allclose(O1, O2)

  def test_user_grad(self):
    self._init_rand()
    E         = self.gen_func_memo()
    G         = E.grad(x=True).proj(1)
    refG      = E._TEST_NIR_Adjoint( x=True )

    I         = self.rand_input()
    Out1      = G(*I)
    Out2      = refG(*(I+(1.0,)))[1]
    np.testing.assert_allclose(Out1, Out2)

  def gen_func(self):
    EPS     = 1e-7

    @ATL.func
    def spring( a : num[2], b : num[2], K_spring : num, link_w : num ):
      disp[k:2] = a[k] - b[k]
      dist      = sqrt(max( EPS, Sum[k:2](disp[k]*disp[k]) ))
      return K_spring * (dist - link_w)

    @ATL.func
    def E_catenary( N : size,
                    link_w : num, K_spring : num, gravity : num,
                    x : num[N,2] ) -> num:
      E_spring  = ( (N>0)*( spring([0,0],x[0],  K_spring,link_w)
                          + spring([1,0],x[N-1],K_spring,link_w) )
                  + Sum[i:N]( (i+1<N)*spring(x[i],x[i+1],K_spring,link_w) ) )
      E_gravity = Sum[i:N]( gravity * x[i,1] )
      return E_spring + E_gravity

    return E_catenary

  def gen_deriv_sig(self):
    return { 'x' : True }

  def gen_deriv(self):
    EPS     = 1e-7
    R2      = num[2]

    @ATL.func
    def Dspring( a:R2, b:R2, da:R2, db:R2, K_spring:num, link_w:num ):
      disp[k:2]   = a[k] - b[k]
      d_disp[k:2] = da[k] - db[k]
      sum_disp    = Sum[k:2](disp[k]*disp[k])
      dist        = sqrt(max( EPS, sum_disp ))
      d_dist      = select_gt( EPS, sum_disp,
                               0, Sum[k:2](disp[k]*d_disp[k]) ) / dist
      return ( K_spring * (dist - link_w),
               K_spring * d_dist )

    @ATL.func
    def deriv_E_catenary(
      N : size,
      link_w : num, K_spring : num, gravity : num,
      x : num[N,2], dx : num[N,2]
    ) -> (num,num):
      Dspring_0   = (N>0)* Dspring([0,0], x[0], [0,0], dx[0],
                                                        K_spring,link_w)
      Dspring_N   = (N>0)* Dspring([1,0], x[N-1], [0,0], dx[N-1],
                                                        K_spring,link_w)
      Dspring_i[i:N]  = (i+1<N) * Dspring(x[i],x[i+1],dx[i],dx[i+1],
                                                        K_spring,link_w)
      E_spring    = ( Dspring_0._0 + Dspring_N._0 +
                      Sum[i:N]( (i+1<N)* Dspring_i[i]._0 ))
      dE_spring   = ( Dspring_0._1 + Dspring_N._1 +
                      Sum[i:N]( (i+1<N)* Dspring_i[i]._1))
      E_gravity   = Sum[i:N]( gravity*x[i,1] )
      dE_gravity  = Sum[i:N]( gravity*dx[i,1] )
      return (E_spring + E_gravity, dE_spring + dE_gravity)

    return deriv_E_catenary
   
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

    return ((N,  link_w, K_spring, gravity,  x),  (dx,))
    
  def rand_deriv_inout(self):
    indata, din   = self.rand_deriv_input()
    w, h          = indata[0:2]
    d_out         = self.rand.uniform(-2,2)
    return (indata,din,d_out)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()

