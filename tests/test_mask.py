from __future__ import annotations

import unittest
from .function_tests import FunctionTestCase
import ATL
from ATL import num

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestMask(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    @ATL.func
    def mask_lapl( w : size, h : size, f : num[h,w], Mask : (h,w) ):
      dx[j:h,i:w]   = (i+1 < w) * (Mask(j,i+1) * (f[j,i+1] - f[j,i]))
      dy[j:h,i:w]   = (j+1 < h) * (Mask(j+1,i) * (f[j+1,i] - f[j,i]))
      return Sum[j:h,i:w]( Mask(j,i)*( dx[j,i]*dx[j,i] + dy[j,i]*dy[j,i] ) )

    return mask_lapl

  def gen_deriv_sig(self):
    return { 'f' : True }

  def gen_deriv(self):
    @ATL.func
    def Dmask_lapl( w : size, h : size, f : num[h,w], Mask : (h,w),
                                       df : num[h,w] ):
      dx[j:h,i:w]   = (i+1 < w) * (Mask(j,i+1) * ( f[j,i+1] - f[j,i]))
      dy[j:h,i:w]   = (j+1 < h) * (Mask(j+1,i) * ( f[j+1,i] - f[j,i]))
      ddx[j:h,i:w]  = (i+1 < w) * (Mask(j,i+1) * (df[j,i+1] - df[j,i]))
      ddy[j:h,i:w]  = (j+1 < h) * (Mask(j+1,i) * (df[j+1,i] - df[j,i]))
      return ( Sum[j:h,i:w](Mask(j,i)*( dx[j,i]*dx[j,i] + dy[j,i]*dy[j,i] )),
               Sum[j:h,i:w](Mask(j,i)*( 2*dx[j,i]*ddx[j,i] +
                                        2*dy[j,i]*ddy[j,i] )) )
    
    return Dmask_lapl

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

  def rand_perf_inout(self):
    w, h      = 1000,1000
    img       = self.rand.rand_ndarray([h,w])
    msk       = self.rand.rand_bool_array([h,w])
    dimg      = self.rand.rand_ndarray([h,w])
    d_out     = self.rand.uniform(-2e3,2e3)
    return ((w,h,img,msk),(dimg,),d_out)


