
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


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()

