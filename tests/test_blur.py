from __future__ import annotations

import unittest
from .function_tests import FunctionTestCase
import ATL
from ATL import num

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestBlur(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    @ATL.func
    def blur( w : size, h : size, img : num[h,w] ):
      blur_x[j:h,i:w]   = 0.25 * ( (i-1 >= 0)*img[j,i-1]
                                  +         2*img[j,i  ]
                                  + (i+1 < w)*img[j,i+1] )
      blur_y[j:h,i:w]   = 0.25 * ( (j-1 >= 0)*blur_x[j-1,i]
                                  +         2*blur_x[j  ,i]
                                  + (j+1 < h)*blur_x[j+1,i] )
      return blur_y

    return blur

  def gen_deriv_sig(self):
    return { 'img' : 'dimg' }

  def gen_deriv(self):
    @ATL.func
    def dblur( w : size, h : size, img : num[h,w], dimg : num[h,w] ):
      blur_x[j:h,i:w]   = 0.25 * ( (i-1 >= 0)*img[j,i-1]
                                  +         2*img[j,i  ]
                                  + (i+1 < w)*img[j,i+1] )
      dblur_x[j:h,i:w]  = 0.25 * ( (i-1 >= 0)*dimg[j,i-1]
                                  +         2*dimg[j,i  ]
                                  + (i+1 < w)*dimg[j,i+1] )
      blur_y[j:h,i:w]   = 0.25 * ( (j-1 >= 0)*blur_x[j-1,i]
                                  +         2*blur_x[j  ,i]
                                  + (j+1 < h)*blur_x[j+1,i] )
      dblur_y[j:h,i:w]  = 0.25 * ( (j-1 >= 0)*dblur_x[j-1,i]
                                  +         2*dblur_x[j  ,i]
                                  + (j+1 < h)*dblur_x[j+1,i] )
      return (blur_y, dblur_y)

    return dblur
    
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
    d_out         = self.rand.rand_ndarray([h,w])
    return (indata,din,d_out)

  def rand_perf_inout(self):
    w, h      = 1000,1000
    img       = self.rand.rand_ndarray([h,w])
    dimg      = self.rand.rand_ndarray([h,w])
    d_out     = self.rand.rand_ndarray([h,w])
    return ((w,h,img),(dimg,),d_out)

  def data_zeros(self):
    w, h      = 4, 4
    indata    = (w,h,np.zeros([h,w],order='F'))
    outdata   = np.zeros([h,w],order='F')
    return indata, outdata

  def data_checker_2(self):
    # a small checker pattern
    w, h      = 8, 6
    img       = np.zeros([h,w],order='F')
    predict   = np.zeros([h,w],order='F')
    for i in range(0,w):
      for j in range(0,h):
        imod  = (i//2) % 2
        jmod  = (j//2) % 2
        val   = 1.0 if imod == jmod else 0.0
        img[j,i] = val
        pval  = (2/16 + 2/8 + 1/4) if val == 1.0 else (2/16 + 2/8)
        # edge correction
        on_ibd  = (i == 0 or i == w-1)
        on_jbd  = (j == 0 or j == h-1)
        if on_ibd:
          if val == 1.0:  pval -= 1/16
          else:           pval -= (1/16 + 1/8)
        if on_jbd:
          if val == 1.0:  pval -= 1/16
          else:           pval -= (1/16 + 1/8)
        if on_ibd and on_jbd:
          if val == 1.0:  pval += 1/16
          else:           pval += 0
        predict[j,i] = pval
    return (w,h,img), predict


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()
