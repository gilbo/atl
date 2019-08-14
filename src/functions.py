
from prelude import *

import atl_types as T
from frontend import UST, AST
from bounds import BoundsExtraction, BoundsCheck

import numpy as np

from py_type_values import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Function:
  def __init__(self,f):
    assert type(f) is UST.function, "do not construct this class directly!"

    # complain about reserved names
    for nm in f.arg_order:
      if nm == 'output':
        raise TypeError(f"The argument name 'output' is reserved, "
                        f"and may not be declared by the user")

    # perform simple type checking and convert to a typed AST
    self._orig_f    = f
    self._ast       = f.typecheck()

    # perform a bounds check on function construction
    bd_sys          = BoundsExtraction(self._ast).system()
    BoundsCheck(bd_sys)

  def __str__(self):
    return str(self._ast)

  def _unpack_call_args(self, *args, **kwargs):
    """ use the named parameter 'output' to specify the output
        "value" (i.e. buffer(s)) or else this function will
        allocate a buffer for output """

    # Step 0: extract output argument, and check that
    #         the correct total number of arguments were provided
    output      = None
    n_supplied  = len(args) + len(kwargs)
    n_expected  = len(self._ast.arg_order)
    if 'output' in kwargs:
      output    = kwargs['output']
      n_supplied -= 1
    if n_supplied != n_expected:
      raise TypeError(f"expected {n_expected} args, but got {n_supplied}")

    # Step 1: convert all arguments into a {Sym : Value} dict
    argdict = {}
    arg_i   = 0
    for idx,sym in enumerate(self._ast.arg_order):
      # case: argument was supplied by name
      if str(sym) in kwargs:
        argdict[sym] = kwargs[str(sym)]
      # case: no more positional arguments available
      elif arg_i >= len(args):
        raise TypeError(f"expected argument '{sym}' but too few "
                        f"unnamed arguments were supplied")
      # case: assume argument was supplied by position
      else:
        argdict[sym] = args[arg_i]
        arg_i += 1

    # Step 2: read those values back out into serialized
    #         vars, sizes, relations arrays for downstream execution
    #         checking types as we go
    vs, szs, rels   = [], [], []
    for sd in self._ast.sizes:
      val   = argdict[sd.name]
      argcheck_python_size(argdict, val, sd.name)
      szs.append(val)
    for vd in self._ast.vars:
      val   = argdict[vd.name]
      argcheck_python_value(argdict, vd.type, val, vd.name)
      vs.append(val)
    for rd in self._ast.relations:
      val   = argdict[rd.name]
      argcheck_python_relation(argdict, val, rd.sizes, rd.name)
      rels.append(val)

    # Step 3: create an output buffer or check the supplied output buffer
    out_type  = self._ast.body.type
    if output is None:
      output  = get_numpy_buffers(argdict, out_type)
    else:
      argcheck_python_value(argdict, out_type, output, 'output',
                            is_output=True)

    return vs, szs, rels, output

  def run_interpreter(self, *args, **kwargs):
    vs, szs, rels, output = self._unpack_call_args(*args,**kwargs)
    # blah






