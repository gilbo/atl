
import unittest
from ATL import *

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestBlur(unittest.TestCase):

  def _gen_func(self):
    num       = Type(float)
    w, h      = Size('w'), Size('h')
    i, j      = IVar('i'), IVar('j')
    img       = Var('img')
    #bd_img    = Var('bd_img')
    blur_x    = Var('blur_x')
    blur_y    = Var('blur_y')

    blur = Fun('blur')[ w, h, img : num[h,w] ]( Let[
      blur_x, Gen[j:h,i:w]( 0.25*( (i-1 >= 0)*img[j,i-1]
                                  +         2*img[j,i  ]
                                  + (i+1 < w)*img[j,i+1] ) ),
      blur_y, Gen[j:h,i:w]( 0.25*( (j-1 >= 0)*blur_x[j-1,i]
                                  +         2*blur_x[j  ,i]
                                  + (j+1 < h)*blur_x[j+1,i] ) ),
    ]( blur_y ))
    return blur

  def test_print_blur(self):
    print( self._gen_func() )

  def test_blur_zeros(self):
    blur      = self._gen_func()

    # a small checker pattern
    w, h      = 4, 4
    img       = np.zeros([h,w])
    predict   = np.zeros([h,w])

    comp      = blur(w,h,img)
    np.testing.assert_allclose(comp, predict)

  def test_blur_checker_2(self):
    blur      = self._gen_func()

    # a small checker pattern
    w, h      = 8, 6
    img       = np.zeros([h,w])
    predict   = np.zeros([h,w])
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

    comp      = blur(w,h,img)
    np.testing.assert_allclose(comp, predict)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()
