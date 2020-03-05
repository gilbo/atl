from __future__ import annotations

import unittest
from .function_tests import FunctionTestCase
import ATL
from ATL import num

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestImgGradient(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    @ATL.func
    def img_grad( w : size, h : size, f : num[h,w] ):
      dx[j:h,i:w] = (i+1 < w) * (f[j,i+1] - f[j,i])
      dy[j:h,i:w] = (j+1 < h) * (f[j+1,i] - f[j,i])
      return (dx,dy)

    return img_grad

  def gen_deriv_sig(self):
    return { 'f' : True }

  def gen_deriv(self):
    @ATL.func
    def Dimg_grad( w : size, h : size, f : num[h,w], df : num[h,w] ):
      dx[j:h,i:w]   = (i+1 < w) * ( f[j,i+1] -  f[j,i])
      dy[j:h,i:w]   = (j+1 < h) * ( f[j+1,i] -  f[j,i])
      ddx[j:h,i:w]  = (i+1 < w) * (df[j,i+1] - df[j,i])
      ddy[j:h,i:w]  = (j+1 < h) * (df[j+1,i] - df[j,i])
      return ((dx,dy),(ddx,ddy))

    return Dimg_grad
    
  def rand_input(self):
    w, h      = self.rand.randint(4,10), self.rand.randint(4,10)
    img       = self.rand.rand_ndarray([h,w])
    return (w,h,img)

  def rand_deriv_input(self):
    w, h, img = self.rand_input()
    dimg      = self.rand.rand_ndarray([h,w])
    return ((w,h,img),(dimg,))

  def rand_deriv_inout(self):
    indata, din   = self.rand_deriv_input()
    w, h          = indata[0:2]
    d_out_x       = self.rand.rand_ndarray([h,w])
    d_out_y       = self.rand.rand_ndarray([h,w])
    return (indata,din,(d_out_x,d_out_y))

  def data_zeros(self):
    w, h      = 4, 4
    img       = np.zeros([h,w],order='F')
    dx_guess  = np.zeros([h,w],order='F')
    dy_guess  = np.zeros([h,w],order='F')
    return (w,h,img), (dx_guess, dy_guess)

  def data_checker_2(self):
    # a small checker pattern
    w, h      = 8, 6
    img       = np.zeros([h,w],order='F')
    dx        = np.zeros([h,w],order='F')
    dy        = np.zeros([h,w],order='F')
    for i in range(0,w):
      for j in range(0,h):
        imod  = (i//2) % 2
        jmod  = (j//2) % 2
        val   = 1.0 if imod == jmod else 0.0
        img[j,i] = val

        # dx
        if i+1 < w and ((i+1)//2)%2 != imod: # change edge
          dx[j,i] = -1.0 if val == 1.0 else 1.0

        # dy
        if j+1 < h and ((j+1)//2)%2 != jmod: # change edge
          dy[j,i] = -1.0 if val == 1.0 else 1.0

    return (w,h,img), (dx,dy)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()


