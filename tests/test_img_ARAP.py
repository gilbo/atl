
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

  def gen_deriv_sig(self):
    return { 'Offsets' : True, 'Angle' : True }

  def gen_deriv(self):
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

    dOffsets      = Var('dOffsets')
    dAngle        = Var('dAngle')

    def DRot2D( ang, v2, dang ):
      s, c  = Var('s'), Var('c')
      scv   = Var('scv')
      cnsv  = Var('cnsv')
      return Let[ s,    ATLmath.sin( ang ),
                  c,    ATLmath.cos( ang ),
                  scv,  s * v2[0] + c * v2[1],
                  cnsv, c * v2[0] - s * v2[1],
                ](([  cnsv, scv  ],
                   [  -scv * dang, cnsv * dang  ]))

    def Dregular(i,j,di,dj):
      d_off, d_ur   = Var('d_off'),  Var('d_ur')
      dd_off        = Var('dd_off')
      Drot, diff    = Var('Drot'), Var('diff')
      ddiff         = Var('ddiff')
      return Let[ d_off,  Gen[k:2]( Offsets[i,j,k]  - Offsets[i+di,j+dj,k] ),
                  dd_off, Gen[k:2]( dOffsets[i,j,k] - dOffsets[i+di,j+dj,k] ),
                  d_ur,   Gen[k:2]( UrShape[i,j,k]  - UrShape[i+di,j+dj,k]  ),
                  Drot,   DRot2D( Angle[i,j], d_ur, dAngle[i,j] ),
                  diff,   Gen[k:2]( d_off[k]  - Drot.proj(0)[k] ),
                  ddiff,  Gen[k:2]( dd_off[k] - Drot.proj(1)[k] ),
              ](( Sum[k:2]( diff[k]*diff[k] ),
                  Sum[k:2]( 2*diff[k]*ddiff[k] ) ))

    E_reg, E_fit    = Var('E_reg'),  Var('E_fit')
    DE_reg, dE_fit  = Var('DE_reg'), Var('dE_fit')
    OC, dOC         = Var('OC'), Var('dOC')

    AoS             = Var('AoS')

    DE_ARAP = Fun('DE_ARAP',(num,num))[
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
      # derivatives
      dOffsets : num[W,H,2], dAngle : num[W,H],
    ](Sum[i:W,j:H]( Mask(i,j) * Let[
        DE_reg, ( (i+1<W) * (Mask(i+1,j) * Dregular(i,j, 1, 0)) +
                 (i-1>=0) * (Mask(i-1,j) * Dregular(i,j,-1, 0)) +
                  (j+1<H) * (Mask(i,j+1) * Dregular(i,j, 0, 1)) +
                 (j-1>=0) * (Mask(i,j-1) * Dregular(i,j, 0,-1)) ),
        OC,     Gen[k:2]( Offsets[i,j,k] - Constraints[i,j,k] ),
        dOC,    dOffsets[i,j],
        E_fit,  Sum[k:2](     OC[k]*OC[k] ),
        dE_fit, Sum[k:2]( 2 * OC[k]*dOC[k] ),
      ]((
        w_fit * C_valid(i,j) * E_fit  + w_reg * DE_reg.proj(0),
        w_fit * C_valid(i,j) * dE_fit + w_reg * DE_reg.proj(1),
      ))))

    return DE_ARAP

  def rand_input(self):
    W,H       = self.rand.randint(5,8), self.rand.randint(5,8)

    w_fit     = self.rand.uniform(0.8,1.2)
    w_reg     = self.rand.uniform(0.8,1.2)

    # unknown transformation
    Offsets   = self.rand.rand_ndarray([W,H,2])
    Angle     = self.rand.rand_ndarray([W,H])

    # known inputs
    UrShape     = self.rand.rand_ndarray([W,H,2])
    Constraints = self.rand.rand_ndarray([W,H,2])

    # masks
    C_valid   = self.rand.rand_bool_array([W,H])
    Mask      = self.rand.rand_bool_array([W,H])

    # zero out C_valid and set Mask to a random circular shape
    m_x, m_y  = self.rand.uniform(W/4,3*W/4), self.rand.uniform(H/4,3*H/4)
    m_r       = self.rand.uniform((W+H)/4,(W+H)/3)
    for i in range(0,W):
      for j in range(0,H):
        C_valid[i,j] = False
        dx,dy = i-m_x, j-m_y
        if dx*dx + dy*dy < m_r*m_r:
          Mask[i,j] = True
        else:
          Mask[i,j] = False

    # set C_valid to a few points now
    N_valid   = self.rand.randint(1,5)
    for k in range(0,N_valid):
      i,j     = self.rand.randint(0,W-1), self.rand.randint(0,H-1)
      C_valid[i,j] = True

    return (W,H, w_fit,w_reg, Offsets,Angle,
            UrShape,Constraints, C_valid,Mask)

  def rand_deriv_input(self):
    (W,H, w_fit,w_reg, Offsets,Angle,
          UrShape,Constraints, C_valid,Mask) = self.rand_input()

    dOffsets  = self.rand.rand_ndarray([W,H,2])
    dAngle    = self.rand.rand_ndarray([W,H])

    return ((W,H, w_fit,w_reg, Offsets,Angle,
                  UrShape,Constraints, C_valid,Mask), (dOffsets,dAngle))

  def rand_deriv_inout(self):
    indata, din   = self.rand_deriv_input()
    W, H          = indata[0:2]
    d_out         = self.rand.uniform(-2,2)
    return (indata,din,d_out)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()


