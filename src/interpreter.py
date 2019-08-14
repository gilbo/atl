
#from adt import ADT
#from adt import memo as ADTmemo

from prelude import *

import atl_types as T
from frontend import AST

from fractions import Fraction

import numpy as np


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #



class _Interpreter:
  def __init__(self, ast, vars, sizes, relations, output):
    self._ctxt    = Context()
    self._env     = Context()









