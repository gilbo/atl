

import unittest
from .function_tests import FunctionTestCase
from ATL import *

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestMask(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    num       = Type(float)
    w, h      = Size('w'), Size('h')
    i, j      = IVar('i'), IVar('j')
    f         = Var('f')
    dx        = Var('dx')
    dy        = Var('dy')
    Mask      = Relation('Mask')

    mask_lapl = Fun('mask_lapl')[ w, h, f : num[h,w], Mask : (h,w) ](
      Let[
        dx, Gen[j:h,i:w]( (i+1 < w) * (Mask(j,i+1) * (f[j,i+1] - f[j,i])) ),
        dy, Gen[j:h,i:w]( (j+1 < h) * (Mask(j+1,i) * (f[j+1,i] - f[j,i])) ),
      ](
        Sum[j:h,i:w]( Mask(j,i)*( dx[j,i] * dx[j,i] + dy[j,i] * dy[j,i] ) )
      ))
    return mask_lapl

#**** s 
# $v_1    = Gen[i0:W, i1:H] [ i0 + -1*i2 == 0, i1 + -1*i3 == 0 ] Angle[i2:W,i3:H]
#$v_2    = Gen[i0:W, i1:H] Sum[i4:W, i5:H]
#          [ i0=i2, i1=i3,
#            1 + i0 + -1*i4 == 0 and i1 + -1*i5 == 0 and Mask([ivar(name=4), ivar(name=5)]) ]
#          sin($v_1)[i2:W,i3:H]
#$v_3    = Gen[i0:W, i1:H] [ i0=i2, i1=i3, -1 + W + -1*i0 > 0 ] $v_2[i2:W,i3:H]
#$v_4    = Gen[i0:W, i1:H] Sum[i4:W, i5:H]
#          [ i0=i2, i1=i3,
#            i0 + -1*i4 == 0 and i1 + -1*i5 == 0 and Mask([ivar(name=4), ivar(name=5)]) ]
#          $v_3[i2:W,i3:H]
#return    $v_4 
#xxxx
# return    0.0


  def gen_deriv_sig(self):
    return { 'f' : True }

  def gen_deriv(self):
    num       = Type(float)
    w, h      = Size('w'), Size('h')
    i, j      = IVar('i'), IVar('j')
    f, df     = Var('f'),  Var('df')
    dx, dy    = Var('dx'), Var('dy')
    ddx, ddy  = Var('ddx'), Var('ddy')
    Mask      = Relation('Mask')

    dmask_lapl = Fun('dmask_lapl')[
      w, h,
      f : num[h,w],
      Mask : (h,w),
      df : num[h,w],
    ]( Let[
      dx,   Gen[j:h,i:w]( (i+1 < w) * (Mask(j,i+1) * (f[j,i+1] - f[j,i])) ),
      dy,   Gen[j:h,i:w]( (j+1 < h) * (Mask(j+1,i) * (f[j+1,i] - f[j,i])) ),
      ddx,  Gen[j:h,i:w]( (i+1 < w) * (Mask(j,i+1) * (df[j,i+1] - df[j,i])) ),
      ddy,  Gen[j:h,i:w]( (j+1 < h) * (Mask(j+1,i) * (df[j+1,i] - df[j,i])) ),
    ](
      ( Sum[j:h,i:w]( Mask(j,i)*(dx[j,i] * dx[j,i] + dy[j,i] * dy[j,i]) ),
        Sum[j:h,i:w]( Mask(j,i)*( 2* dx[j,i] * ddx[j,i] +
                                  2 * dy[j,i] * ddy[j,i] ) )
      )
    ))
    return dmask_lapl


  def rand_input(self):
    w, h      = self.rand.randint(10,20), self.rand.randint(10,20)
    img       = self.rand.rand_ndarray([h,w])
    msk       = self.rand.rand_bool_array([h,w])
    return (w,h,img,msk)

  def rand_deriv_input(self):
    w, h, img, msk  = self.rand_input()
    dimg            = self.rand.rand_ndarray([h,w])
    return ((w,h,img,msk),(dimg,))

  def rand_deriv_inout(self):
    indata, din   = self.rand_deriv_input()
    w, h          = indata[0:2]
    d_out         = self.rand.uniform(-2,2)
    return (indata,din,d_out)


