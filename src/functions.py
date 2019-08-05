
from prelude import *

import atl_types as T
from frontend import UST, AST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Function:
  def __init__(self,f):
    assert type(f) is UST.function, "do not construct this class directly!"

    self._orig_f    = f
    self._ast       = f.typecheck()

  def __str__(self):
    return str(self._ast)




