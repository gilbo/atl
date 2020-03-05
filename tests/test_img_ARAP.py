from __future__ import annotations

import unittest
from .function_tests import FunctionTestCase
import ATL
from ATL import num, Type

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestImgARAP(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    @ATL.func
    def Rot2D( ang : num, v2 : num[2] ):
      s   = sin(ang)
      c   = cos(ang)
      return [ c*v2[0] - s*v2[1],
               s*v2[0] + c*v2[1] ]

    @ATL.macro
    def regular(i,j,di,dj,P):
      d_off[k:2]  = P.Offsets[i,j,k] - P.Offsets[i+di,j+dj,k]
      d_ur[k:2]   = P.UrShape[i,j,k] - P.UrShape[i+di,j+dj,k]
      rot         = Rot2D( P.Angle[i,j], d_ur )
      diff[k:2]   = d_off[k] - rot[k]
      return Sum[k:2]( diff[k]*diff[k] )

    @ATL.func
    def E_ARAP(
      # image size
      W : size, H : size,
      # energy term weights
      w_fit : num, w_reg : num,
      # unknown vector fields
      Offsets : num[W,H,2], Angle : num[W,H],
      # input (known) vector fields
      UrShape : num[W,H,2], Constraints : num[W,H,2],
      # masking relations
      C_valid : (W,H), Mask : (W,H)
    ):
      P               = { Offsets:Offsets, UrShape:UrShape, Angle:Angle }
      E_reg[i:W,j:H]  = ( (i+1<W) * (Mask(i+1,j) * regular(i,j, 1, 0, P)) +
                         (i-1>=0) * (Mask(i-1,j) * regular(i,j,-1, 0, P)) +
                          (j+1<H) * (Mask(i,j+1) * regular(i,j, 0, 1, P)) +
                         (j-1>=0) * (Mask(i,j-1) * regular(i,j, 0,-1, P)) )
      OC[i:W,j:H,k:2] = Offsets[i,j,k] - Constraints[i,j,k]
      E_fit[i:W,j:H]  = Sum[k:2]( OC[i,j,k]*OC[i,j,k] )
      return Sum[i:W,j:H](Mask(i,j)*(
                            w_fit * (C_valid(i,j) * E_fit[i,j])
                          + w_reg * E_reg[i,j] ))
    
    return E_ARAP

  def gen_deriv_sig(self):
    return { 'Offsets' : True, 'Angle' : True }

  def gen_deriv(self):
    @ATL.func
    def DRot2D( ang : num, v2 : num[2], dang : num ):
      s     = sin(ang)
      c     = cos(ang)
      scv   = s*v2[0] + c*v2[1]
      cnsv  = c*v2[0] - s*v2[1]
      return ([ cnsv, scv ],
              [ -1.0 * scv * dang, cnsv * dang ])

    @ATL.macro
    def Dregular(i,j,di,dj,P):
      d_off[k:2]  = P.Offsets[i,j,k]  - P.Offsets[i+di,j+dj,k]
      dd_off[k:2] = P.dOffsets[i,j,k] - P.dOffsets[i+di,j+dj,k]
      d_ur[k:2]   = P.UrShape[i,j,k]  - P.UrShape[i+di,j+dj,k]
      Drot        = DRot2D( P.Angle[i,j], d_ur, P.dAngle[i,j] )
      diff[k:2]   = d_off[k]  - (Drot._0)[k]
      ddiff[k:2]  = dd_off[k] - (Drot._1)[k]
      return ( Sum[k:2]( diff[k]*diff[k] ),
               Sum[k:2]( 2*diff[k]*ddiff[k] ) )

    @ATL.func
    def DE_ARAP(
      # image size
      W : size, H : size,
      # energy term weights
      w_fit : num, w_reg : num,
      # unknown vector fields
      Offsets : num[W,H,2], Angle : num[W,H],
      # input (known) vector fields
      UrShape : num[W,H,2], Constraints : num[W,H,2],
      # masking relations
      C_valid : (W,H), Mask : (W,H),
      # derivatives
      dOffsets : num[W,H,2], dAngle : num[W,H]
    ):
      P               = { Offsets:Offsets, UrShape:UrShape, Angle:Angle,
                          dOffsets:dOffsets, dAngle:dAngle }
      DE_reg[i:W,j:H] = ( (i+1<W) * (Mask(i+1,j) * Dregular(i,j, 1, 0, P)) +
                         (i-1>=0) * (Mask(i-1,j) * Dregular(i,j,-1, 0, P)) +
                          (j+1<H) * (Mask(i,j+1) * Dregular(i,j, 0, 1, P)) +
                         (j-1>=0) * (Mask(i,j-1) * Dregular(i,j, 0,-1, P)) )
      OC[i:W,j:H,k:2]   = Offsets[i,j,k] - Constraints[i,j,k]
      dOC[i:W,j:H,k:2]  = dOffsets[i,j,k]
      E_fit[i:W,j:H]    = Sum[k:2](   OC[i,j,k]*OC[i,j,k] )
      dE_fit[i:W,j:H]   = Sum[k:2]( 2*OC[i,j,k]*dOC[i,j,k] )
      return ( Sum[i:W,j:H](Mask(i,j)*(
                              w_fit * (C_valid(i,j) * E_fit[i,j])
                            + w_reg * DE_reg[i,j]._0 )),
               Sum[i:W,j:H](Mask(i,j)*(
                              w_fit * (C_valid(i,j) * dE_fit[i,j])
                            + w_reg * DE_reg[i,j]._1 )) )
    
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


