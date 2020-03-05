from __future__ import annotations

import unittest
from .function_tests import FunctionTestCase
import ATL
from ATL import num

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestImgLaplacian(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    @ATL.func
    def img_laplacian( w : size, h : size, f : num[h,w] ):
      dx[j:h,i:w]   = (i+1 < w) * (f[j,i+1] - f[j,i])
      dy[j:h,i:w]   = (j+1 < h) * (f[j+1,i] - f[j,i])
      return Sum[j:h,i:w]( dx[j,i] * dx[j,i] + dy[j,i] * dy[j,i] )

    return img_laplacian

  # btw, a nicer syntax for the above would be something like
  # @ATL.function
  # def img_laplacian( w:size, h:size, f : num[h,w] ):
  #   dx[j,i] = (i+1 < w) * ( f[j,i+1] - f[j,i] )
  #   dy[j,i] = (j+1 < h) * ( f[j+1,i] - f[j,i] )
  #   return Sum[j,i]( dx[j,i]**2 + dy[j,i]**2 )
  #
  # basically have this now, except exponentiation

  def gen_deriv_sig(self):
    return { 'f' : True }

  def gen_deriv(self):
    @ATL.func
    def Dimg_laplacian( w : size, h : size, f : num[h,w], df : num[h,w] ):
      dx[j:h,i:w]   = (i+1 < w) * ( f[j,i+1] -  f[j,i])
      dy[j:h,i:w]   = (j+1 < h) * ( f[j+1,i] -  f[j,i])
      ddx[j:h,i:w]  = (i+1 < w) * (df[j,i+1] - df[j,i])
      ddy[j:h,i:w]  = (j+1 < h) * (df[j+1,i] - df[j,i])
      return ( Sum[j:h,i:w](  dx[j,i] * dx[j,i] +  dy[j,i] * dy[j,i] ),
               Sum[j:h,i:w]( 2*dx[j,i]*ddx[j,i] + 2*dy[j,i]*ddy[j,i] ) )

    return Dimg_laplacian

  def rand_input(self):
    w, h      = self.rand.randint(10,20), self.rand.randint(10,20)
    img       = self.rand.rand_ndarray([h,w])
    return (w,h,img)

  def rand_deriv_input(self):
    w, h, img = self.rand_input()
    dimg      = self.rand.rand_ndarray([h,w])
    return ((w,h,img),(dimg,))

  def rand_deriv_inout(self):
    indata, din   = self.rand_deriv_input()
    w, h          = indata[0:2]
    d_out         = self.rand.uniform(-2,2)
    return (indata,din,d_out)
    
  def data_zeros(self):
    w, h      = 4, 4
    img       = np.zeros([h,w], order='F')
    predict   = 0.0
    return (w,h,img), predict

  def data_checker_2(self):
    # a small checker pattern
    w, h      = 8, 6
    img       = np.zeros([h,w], order='F')
    predict   = float( 3*6 + 8*2 )
    
    for i in range(0,w):
      for j in range(0,h):
        imod  = (i//2) % 2
        jmod  = (j//2) % 2
        val   = 1.0 if imod == jmod else 0.0
        img[j,i] = val

    return (w,h,img), predict



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()



