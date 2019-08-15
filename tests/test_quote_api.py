
import unittest
from ATL.quote_api import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestQuoteAPI(unittest.TestCase):

  def test_dot_shorthand(self):
    x, y  = Var('x'), Var('y')
    i     = IVar('i')
    n     = Size('n')
    num   = Type(float)

    f = Fun('dot')[ n, x : num[n], y : num[n] ](
      Sum[i:n]( x[i] * ((i+1 < n) * y[i+1]) )
    )

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()
