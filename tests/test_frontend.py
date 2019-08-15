
import unittest
from ATL.frontend import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TestFrontend(unittest.TestCase):

  def test_dot_ust(self):
    x   = 'x'
    y   = 'y'
    i   = 'i'
    n   = Sym('n')
    NS  = null_srcinfo()

    body = UST.Sum(i,n,
              UST.Mul( UST.Access(UST.Var(x,NS),
                                  [UST.IdxVar(i,NS)], NS),
                       UST.Access(UST.Var(y,NS),
                                  [UST.IdxVar(i,NS)], NS), NS ), NS)

    f = UST.function( name  = 'dot',
                      arg_order = ['n','x','y'],
                      rettype   = T.num,
                      vars  = [UST.var_decl(x, T.Tensor(n,T.num), NS),
                               UST.var_decl(y, T.Tensor(n,T.num), NS)],
                      sizes = [UST.size_decl(n, NS)],
                      relations = [],
                      body  = body,
                      srcinfo   = NS )

    ast = f.typecheck()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()
