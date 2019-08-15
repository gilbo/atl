
import unittest
from ATL import *

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestImgGradient(unittest.TestCase):

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

  def test_print_img_gradient(self):
    print(self.gen_func())


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()


