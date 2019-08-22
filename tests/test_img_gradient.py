
import unittest
from .function_tests import FunctionTestCase
from ATL import *

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestImgGradient(unittest.TestCase, FunctionTestCase):

  def gen_func(self):
    num       = Type(float)
    w, h      = Size('w'), Size('h')
    i, j      = IVar('i'), IVar('j')
    f         = Var('f')
    dx        = Var('dx')
    dy        = Var('dy')

    img_grad = Fun('img_grad')[ w, h, f : num[h,w] ]( Let[
      dx, Gen[j:h,i:w]( (i+1 < w) * (f[j,i+1] - f[j,i]) ),
      dy, Gen[j:h,i:w]( (j+1 < h) * (f[j+1,i] - f[j,i]) ),
    ]( (dx,dy) ))
    return img_grad

  def data_zeros(self):
    w, h      = 4, 4
    img       = np.zeros([h,w])
    dx_guess  = np.zeros([h,w])
    dy_guess  = np.zeros([h,w])
    return (w,h,img), (dx_guess, dy_guess)

  def data_checker_2(self):
    # a small checker pattern
    w, h      = 8, 6
    img       = np.zeros([h,w])
    dx        = np.zeros([h,w])
    dy        = np.zeros([h,w])
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


