
from .prelude import *

from . import atl_types as T
from . import frontend as F

#import numpy as np

import math

#from .py_type_values import *

#from .interpreter import Interpret

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class BuiltIn_Typecheck_Error(Exception):
  def __init__(self,msg):
    self._builtin_err_msg   = str(msg)
  def __str__(self):
    return self._builtin_err_msg

_BErr = BuiltIn_Typecheck_Error

class BuiltIn:
  def __init__(self,name):
    self._name  = name

  def name(self):
    return self._name

  def typecheck(self,*args):
    raise NotImplementedError()

  def deriv(self,*args):
    raise NotImplementedError()

  def interpret(self,*args):
    raise NotImplementedError()


class ScalarBI(BuiltIn):
  def __init__(self,name,n_ary):
    self._n_ary = n_ary
    super().__init__(name)

  def typecheck(self,*args):
    if len(args) != self._n_ary:
      raise _BErr(f"expected {self._n_ary} argument(s), got {len(args)}")
    result = T.error
    for k,a in enumerate(args):
      if a is T.error:
        pass
      elif a is not T.num:
        raise _BErr(f"expected argument {k} to have type Num, got {a}")
    return T.num


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class _Sin(ScalarBI):
  def __init__(self):
    super().__init__('sin',1)

  def deriv( self, x, dx, srcinfo=null_srcinfo() ):
    cos_x       = F.AST.BuiltIn(cos, [x], T.num, srcinfo)
    return F.AST.BinOp( '*', cos_x, dx, T.num, srcinfo )

  def interpret(self,x):
    return math.sin(x)
sin = _Sin()

class _Cos(ScalarBI):
  def __init__(self):
    super().__init__('cos',1)

  def deriv( self, x, dx, srcinfo=null_srcinfo() ):
    neg_sin_x   = F.AST.BinOp( '*', F.AST.Const(-1.0, T.num, srcinfo),
                                    F.AST.BuiltIn(sin, [x], T.num, srcinfo),
                                    T.num, srcinfo )
    return F.AST.BinOp( '*', neg_sin_x, dx, T.num, srcinfo )

  def interpret(self,x):
    return math.cos(x)
cos = _Cos()

class _Sqrt(ScalarBI):
  def __init__(self):
    super().__init__('sqrt',1)

  def deriv( self, x, dx, srcinfo=null_srcinfo() ):
    return F.AST.BinOp( '/',
                        F.AST.BinOp( '*', F.AST.Const(0.5, T.num, srcinfo),
                                          dx, T.num, srcinfo ),
                        F.AST.BuiltIn(sqrt, [x], T.num, srcinfo),
                        T.num, srcinfo )

  def interpret(self,x):
    return math.sqrt(x)
sqrt = _Sqrt()

class _Ln(ScalarBI):
  def __init__(self):
    super().__init__('ln',1)

  def deriv( self, x, dx, srcinfo=null_srcinfo() ):
    return F.AST.BinOp( '/', dx, x, T.num, srcinfo )

  def interpret(self,x):
    return math.log(x)
ln = _Ln()

class _Pow(ScalarBI):
  def __init__(self):
    super().__init__('pow',2)

  def deriv( self, x, y, dx, dy, srcinfo=null_srcinfo() ):
    # for efficiency, bind x and y as temporaries
    base, exp     = Sym('base'), Sym('exp')
    xstmt         = F.AST.assign( base, T.num, x, srcinfo )
    ystmt         = F.AST.assign( exp,  T.num, y, srcinfo )
    x, y          = F.AST.Var(base), F.AST.Var(exp)

    # D[[ x^y | x ]] =    y * x^(y-1) * dx
    # D[[ x^y | y ]] =  x^y *  ln(x)  * dy
    y_n1          = F.AST.BinOp("-",y, F.AST.Const(1.0,T.num,srcinfo),
                                T.num, srcinfo)
    pow_dx        = F.AST.BinOp("*", y,
                                F.AST.BuiltIn(pow,[x,y_n1],T.num,srcinfo),
                                T.num, srcinfo)
    pow_dy        = F.AST.BinOp("*",
                                F.AST.BuiltIn(pow,[x,y],T.num,srcinfo),
                                F.AST.BuiltIn(ln,[x],T.num,srcinfo),
                                T.num, srcinfo)

    return F.AST.Let( [xstmt, ystmt],
              F.AST.BinOp("+",
                  F.AST.BinOp("*", pow_dx, dx, T.num, srcinfo),
                  F.AST.BinOp("*", pow_dy, dy, T.num, srcinfo),
                  T.num, srcinfo),
              T.num, srcinfo)

  def interpret(self,x,y):
    return x ** y
pow = _Pow()

class _Select_GT(ScalarBI):
  def __init__(self):
    super().__init__('select_gt',4)

  def deriv( self, x, y, a, b, dx, dy, da, db, srcinfo=null_srcinfo() ):
    return F.AST.BuiltIn( select_gt, [x, y, da, db], T.num, srcinfo )

  def interpret(self,x,y,a,b):
    return a if x > y else b
select_gt = _Select_GT()

class _Max(ScalarBI):
  def __init__(self):
    super().__init__('max',2)

  def deriv( self, a, b, da, db, srcinfo=null_srcinfo() ):
    return F.AST.BuiltIn( select_gt, [a, b, da, db], T.num, srcinfo )

  def interpret(self,a,b):
    return max(a,b)
fmax = _Max()

class _Min(ScalarBI):
  def __init__(self):
    super().__init__('min',2)

  def deriv( self, a, b, da, db, srcinfo=null_srcinfo() ):
    neg_a   = F.AST.BinOp( '*', F.AST.Const(-1.0, T.num, srcinfo),
                                a, T.num, srcinfo )
    neg_b   = F.AST.BinOp( '*', F.AST.Const(-1.0, T.num, srcinfo),
                                b, T.num, srcinfo )
    return F.AST.BuiltIn( select_gt, [neg_a, neg_b, da, db], T.num, srcinfo )

  def interpret(self,a,b):
    return min(a,b)
fmin = _Min()



