
from .prelude import *

from . import atl_types as T
#from .frontend import UST, AST

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
    self._name = name

  def name(self):
    return self._name

  def typecheck(self,*args):
    raise NotImplementedError()

  def interpret(self,*args):
    raise NotImplementedError()




class _Sin(BuiltIn):
  def __init__(self):
    super().__init__('sin')

  def typecheck(self,*args):
    if len(args) != 1:
      raise _BErr(f"expected 1 argument, got {len(args)}")
    if args[0] is T.error:
      return T.error
    elif args[0] is not T.num:
      raise _BErr(f"expected an argument of type Num, got {args[0]}")
    return T.num

  def interpret(self,x):
    return math.sin(x)
sin = _Sin()

class _Cos(BuiltIn):
  def __init__(self):
    super().__init__('cos')

  def typecheck(self,*args):
    if len(args) != 1:
      raise _BErr(f"expected 1 argument, got {len(args)}")
    if args[0] is T.error:
      return T.error
    elif args[0] is not T.num:
      raise _BErr(f"expected an argument of type Num, "
                  f"got {args[0]}")
    return T.num

  def interpret(self,x):
    return math.cos(x)
cos = _Cos()

class _Sqrt(BuiltIn):
  def __init__(self):
    super().__init__('sqrt')

  def typecheck(self,*args):
    if len(args) != 1:
      raise _BErr(f"expected 1 argument, got {len(args)}")
    if args[0] is T.error:
      return T.error
    elif args[0] is not T.num:
      raise _BErr(f"expected an argument of type Num, "
                  f"got {args[0]}")
    return T.num

  def interpret(self,x):
    return math.sqrt(x)
sqrt = _Sqrt()

class _Max(BuiltIn):
  def __init__(self):
    super().__init__('max')

  def typecheck(self,*args):
    if len(args) != 2:
      raise _BErr(f"expected 2 arguments, got {len(args)}")
    for i,a in enumerate(args):
      if a is not T.error and a is not T.num:
        raise _BErr(f"expected argument {i} to have type Num, got {a}")
    if args[0] is T.error or args[1] is T.error:
      return T.error
    else:
      return T.num

  def interpret(self,a,b):
    return max(a,b)
fmax = _Max()



