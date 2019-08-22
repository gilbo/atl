
import unittest
from .function_tests import FunctionTestCase
from ATL import *

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestImgARAP(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    num       = Type(float)
    W, H      = Size('W'), Size('H')
    w_fit, w_reg  = Var('w_fit'), Var('w_reg')
    Offsets       = Var('Offsets')
    Angle         = Var('Angle')
    UrShape       = Var('UrShape')
    Constraints   = Var('Constraints')
    C_valid       = Relation('C_valid')
    Mask          = Relation('Mask')
    i, j      = IVar('i'), IVar('j')
    k         = IVar('k')

    def Rot2D( ang, v2 ):
      s, c  = Var('s'), Var('c')
      return Let[ s, ATLmath.sin( ang ),
                  c, ATLmath.cos( ang ),
                ]([  c * v2[0] - s * v2[1],
                     s * v2[0] + c * v2[1]  ])

    def regular(i,j,di,dj):
      d_off, d_ur   = Var('d_off'), Var('d_ur')
      rot, diff     = Var('rot'), Var('diff')
      return Let[ d_off,  Gen[k:2]( Offsets[i,j,k] - Offsets[i+di,j+dj,k] ),
                  d_ur,   Gen[k:2]( UrShape[i,j,k] - UrShape[i+di,j+dj,k] ),
                  rot,    Rot2D( Angle[i,j], d_ur ),
                  diff,   Gen[k:2]( d_off[k] - rot[k] ),
             ]( Sum[k:2]( diff[k]*diff[k] ) )

    E_reg, E_fit  = Var('E_reg'), Var('E_fit')
    OC            = Var('OC')

    E_ARAP = Fun('E_ARAP',num)[
      # image size
      W, H,
      # energy term weights
      w_fit : num, w_reg : num,
      # unknown vector fields
      Offsets : num[W,H,2], Angle : num[W,H],
      # input (known) vector fields
      UrShape : num[W,H,2], Constraints : num[W,H,2],
      # masking relations
      C_valid : (W,H), Mask : (W,H),
    ](
      Sum[i:W,j:H]( Mask(i,j) * Let[
        E_reg, ( (i+1<W) * (Mask(i+1,j) * regular(i,j, 1, 0)) +
                (i-1>=0) * (Mask(i-1,j) * regular(i,j,-1, 0)) +
                 (j+1<H) * (Mask(i,j+1) * regular(i,j, 0, 1)) +
                (j-1>=0) * (Mask(i,j-1) * regular(i,j, 0,-1)) ),
        OC,     Gen[k:2]( Offsets[i,j,k] - Constraints[i,j,k] ),
        E_fit,  Sum[k:2]( OC[k]*OC[k] ),
      ](  w_fit * C_valid(i,j) * E_fit + w_reg * E_reg  ))
    )

    return E_ARAP


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()


