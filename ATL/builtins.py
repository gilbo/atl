
from .prelude import *

from . import atl_types as T
from . import frontend as F
from . import norm_ir as N
from .halide_ir import HIR

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

  def nir_deriv(self,*args):
    raise NotImplementedError()

  def nir_adjoint(self,*args):
    raise NotImplementedError()

  def halide_compile(self,*args):
    raise NotImplementedError()

  def C_compile(self,*args):
    raise NotImplementedError()

  def interpret(self,*args):
    raise NotImplementedError()

  # determine whether 0-valued inputs can cause seg-faults
  def crash_on_zero(self):
    raise NotImplementedError()

  # optionally can implement some kind of simplification...
  def simplify(self,orig_node,args):
    return N.NIR.BuiltIn( orig_node.f, args, orig_node.type )

class ScalarBI(BuiltIn):
  def __init__(self,name,n_ary):
    self._n_ary = n_ary
    super().__init__(name)

  # by default assume that scalar built-ins are zero-insensitive
  def crash_on_zero(self):
    return False

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

  def simplify(self,orig_node,args):
    if all([ type(a) is N.NIR.Const for a in args ]):
      # implement constant propagation?
      val = self.interpret( *[ a.val for a in args ] )
      return N.NIR.Const( val, orig_node.type )
    else:
      return super().simplify(orig_node,args)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class _Sin(ScalarBI):
  def __init__(self):
    super().__init__('sin',1)

  def deriv( self, x, dx, srcinfo=null_srcinfo() ):
    cos_x       = F.AST.BuiltIn(cos, [x], T.num, srcinfo)
    return F.AST.BinOp( '*', cos_x, dx, T.num, srcinfo )

  def nir_deriv( self, x, dx ):
    cos_x       = N.NIR.BuiltIn(cos, [x], x.type)
    return N.nir_mul(cos_x, dx)

  def nir_adjoint( self, diffT, x ):
    cos_x       = N.NIR.BuiltIn(cos, [x], x.type)
    dx          = N.nir_mul(cos_x, diffT)
    return (dx,)

  def halide_compile( self, x ):
    return HIR.MathFn1("sin", x)

  def C_compile(self, x):
    return f"sin({x})"

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

  def nir_deriv( self, x, dx ):
    sin_x       = N.NIR.BuiltIn(sin, [x], x.type)
    return N.nir_mul(sin_x, dx, coeff=-1.0)

  def nir_adjoint( self, diffT, x ):
    sin_x       = N.NIR.BuiltIn(sin, [x], x.type)
    dx          = N.nir_mul(sin_x, diffT, coeff=-1.0)
    return (dx,)

  def halide_compile( self, x ):
    return HIR.MathFn1("cos", x)

  def C_compile(self, x):
    return f"cos({x})"

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

  def nir_deriv( self, x, dx ):
    assert False, "Should have been converted to NIR.Pow"

  def nir_adjoint( self, diffT, x ):
    assert False, "Should have been converted to NIR.Pow"

  def halide_compile( self, x ):
    return HIR.MathFn1("sqrt", x)

  def C_compile(self, x):
    return f"sqrt({x})"

  def interpret(self,x):
    return math.sqrt(x)
sqrt = _Sqrt()

class _Ln(ScalarBI):
  def __init__(self):
    super().__init__('ln',1)

  # definitely undefined on zero
  def crash_on_zero(self):
    return True

  def deriv( self, x, dx, srcinfo=null_srcinfo() ):
    return F.AST.BinOp( '/', dx, x, T.num, srcinfo )

  def nir_deriv( self, x, dx ):
    inv_x       = N.NIR.Pow(x, Fraction(-1), x.type)
    return N.nir_mul(dx, inv_x)

  def nir_adjoint( self, diffT, x ):
    inv_x       = N.NIR.Pow(x, Fraction(-1), x.type)
    dx          = N.nir_mul(inv_x, diffT)
    return (dx,)

  def halide_compile( self, x ):
    return HIR.MathFn1("log", x)

  def C_compile(self, x):
    return f"ln({x})"

  def interpret(self,x):
    return math.log(x)
ln = _Ln()

class _Pow(ScalarBI):
  def __init__(self):
    super().__init__('pow',2)

  # if the exponent is negative, then we can cause a divide by zero
  def crash_on_zero(self):
    return True

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

  def nir_deriv( self, x, y, dx, dy ):
    raise NotImplementedError("need to handle exponents")

  def nir_adjoint( self, diffT, x, y ):
    raise NotImplementedError("need to handle exponents")

  def halide_compile( self, x, y ):
    return HIR.Pow(x,y)

  def C_compile(self, x, y):
    return f"pow({x},{y})"

  def interpret(self,x,y):
    return x ** y
pow = _Pow()

class _Select_GT(ScalarBI):
  def __init__(self):
    super().__init__('select_gt',4)

  def deriv( self, x, y, a, b, dx, dy, da, db, srcinfo=null_srcinfo() ):
    return F.AST.BuiltIn( select_gt, [x, y, da, db], T.num, srcinfo )

  def nir_deriv( self, x, y, a, b, dx, dy, da, db ):
    return N.NIR.BuiltIn( select_gt, [x,y,da,db], x.type)

  def nir_adjoint( self, diffT, x, y, a, b ):
    zero    = N.NIR.Const( 0.0, x.type )
    da      = N.NIR.BuiltIn( select_gt, [x,y,diffT,zero], x.type )
    db      = N.NIR.BuiltIn( select_gt, [x,y,zero,diffT], x.type )
    return zero, zero, da, db

  def halide_compile( self, x, y, a, b ):
    return HIR.Select( HIR.BinOp(">=",x,y), a, b )

  def C_compile(self, x, y, a, b):
    return f"(({x} >= {y})? {a} : {b})"

  def interpret(self,x,y,a,b):
    return a if x > y else b
select_gt = _Select_GT()

class _Max(ScalarBI):
  def __init__(self):
    super().__init__('max',2)

  def deriv( self, a, b, da, db, srcinfo=null_srcinfo() ):
    return F.AST.BuiltIn( select_gt, [a, b, da, db], T.num, srcinfo )

  def nir_deriv( self, a, b, da, db ):
    return N.NIR.BuiltIn( select_gt, [a,b,da,db], a.type)

  def nir_adjoint( self, diffT, a, b ):
    zero    = N.NIR.Const( 0.0, a.type )
    da      = N.NIR.BuiltIn( select_gt, [a,b,diffT,zero], a.type )
    db      = N.NIR.BuiltIn( select_gt, [a,b,zero,diffT], a.type )
    return da, db

  def halide_compile( self, a, b ):
    return HIR.Max( a,b )

  def C_compile(self, a, b):
    return f"fmax({a},{b})"

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

  def nir_deriv( self, a, b, da, db ):
    neg_1   = N.NIR.Const(1.0,a.type)
    neg_a   = N.nir_mul(neg_1, a)
    neg_b   = N.nir_mul(neg_1, b)
    return N.NIR.BuiltIn( select_gt, [neg_a,neg_b,da,db], a.type )

  def nir_adjoint( self, diffT, a, b ):
    neg_1   = N.NIR.Const(1.0,a.type)
    neg_a   = N.nir_mul(neg_1, a)
    neg_b   = N.nir_mul(neg_1, b)
    zero    = N.NIR.Const( 0.0, a.type )
    da      = N.NIR.BuiltIn( select_gt, [neg_a,neg_b,diffT,zero], a.type )
    db      = N.NIR.BuiltIn( select_gt, [neg_a,neg_b,zero,diffT], a.type )
    return da, db

  def halide_compile( self, a, b ):
    return HIR.Min( a,b )

  def C_compile(self, a, b):
    return f"fmin({a},{b})"

  def interpret(self,a,b):
    return min(a,b)
fmin = _Min()



