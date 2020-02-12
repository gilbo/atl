
from .prelude import *

from . import atl_types as T
from .frontend import UST, AST
from .bounds import BoundsExtraction, BoundsCheck

import numpy as np

from .py_type_values import *

from .interpreter import Interpret
from .ast_to_halide import Compile as HCompile
from .perf_analysis import Analysis
from .norm_ast    import LetLift, TupleElimination, IndexDownGenUp
from .deriv_ast   import TotalDerivative
from .norm_ir     import AST_to_NIR, NIR_to_AST
from .nir_deriv   import NIR_Deriv


from .checks      import TC_Lite

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Function:
  def __init__(self, f, _do_bound_check=True):
    if ( type(f) is not UST.function and
         type(f) is not AST.function ):
      raise TypeError("do not construct Function objects directly!")

    # complain about reserved names
    for nm in f.arg_order:
      if str(nm) == 'output':
        raise TypeError(f"The argument name 'output' is reserved, "
                        f"and may not be declared by the user")

    # perform simple type checking and convert to a typed AST
    if type(f) is UST.function:
      f             = f.typecheck()
    self._ast       = f
    self._test_arg_order_consistency(self._ast)

    # perform a bounds check on function construction
    if _do_bound_check:
      bd_sys        = BoundsExtraction(self._ast).system()
      BoundsCheck(bd_sys)

  def __str__(self):
    return str(self._ast)

  def _test_arg_order_consistency(self, ast):
    vi, si, ri      = 0, 0, 0
    for i,nm in enumerate(ast.arg_order):
      if   vi < len(ast.vars) and ast.vars[vi].name == nm:
        vi += 1
      elif si < len(ast.sizes) and ast.sizes[si].name == nm:
        si += 1
      elif ri < len(ast.relations) and ast.relations[ri].name == nm:
        ri += 1
      else: assert False, "bad name"
    assert vi == len(ast.vars)
    assert si == len(ast.sizes)
    assert ri == len(ast.relations)

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
      #print('SZ ', sd.name, type(val))
      argcheck_python_size(argdict, val, sd.name)
      szs.append(val)
    for vd in self._ast.vars:
      val   = argdict[vd.name]
      #print('VAR', vd.name, type(val))
      argcheck_python_value(argdict, vd.type, val, vd.name)
      vs.append(val)
    for rd in self._ast.relations:
      val   = argdict[rd.name]
      #print('REL', rd.name, type(val))
      argcheck_python_relation(argdict, rd.sizes, val, rd.name)
      rels.append(val)

    # Step 3: create an output buffer or check the supplied output buffer
    out_type  = self._ast.body.type
    if output is None:
      output  = get_numpy_buffers(argdict, out_type)
    else:
      argcheck_python_value(argdict, out_type, output, 'output',
                            is_output=True)

    return vs, szs, rels, output

  def _pack_return_scalars(self,typ,output):
    if typ is T.num:
      return output[0]
    elif type(typ) is T.Tuple:
      ctr = get_python_named_tuple(typ)
      return ctr(*[ self._pack_return_scalars(t,o)
                    for t,o in zip(typ.types, output) ])
    elif type(typ) is T.Tensor:
      return output

  def interpret(self, *args, **kwargs):
    vs, szs, rels, output = self._unpack_call_args(*args,**kwargs)
    Interpret(self._ast, vs, szs, rels, output)
    return self._pack_return_scalars(self._ast.rettype, output)

  def jit_exec(self, *args, **kwargs):
    vs, szs, rels, output = self._unpack_call_args(*args,**kwargs)
    if not hasattr(self, '_jit_halide_compiled'):
      #print(self._prenorm_ast())
      self._jit_halide_compiled = HCompile(self._prenorm_ast())
    self._jit_halide_compiled(vs, szs, rels, output)
    return self._pack_return_scalars(self._ast.rettype, output)

  def __call__(self, *args, **kwargs):
    return self.jit_exec(*args,**kwargs)
    #return self.interpret(*args,**kwargs)

  def perf_counts(self, *args, **kwargs):
    vs, szs, rels, output = self._unpack_call_args(*args,**kwargs)
    Analysis(self._ast, vs, szs, rels)

  def _prenorm_ast(self):
    if not hasattr(self, '_prenorm_ast_cached'):
      ast           = self._ast
      ast           = LetLift(self._ast).normalized()
      ast           = TupleElimination(ast).normalized()
      ast           = IndexDownGenUp(ast).normalized()
      self._prenorm_ast_cached = ast
    return self._prenorm_ast_cached

  def _unpack_deriv_args(self, *args, **kwargs):
    """ use the named parameter 'output' to name the output
        or else this function will default to 'out' """

    # Step 0: extract output name if present, and check that
    #         not too many arguments were supplied.
    output      = None
    n_supplied  = len(args) + len(kwargs)
    n_expected  = len(self._ast.vars)
    if 'output' in kwargs:
      output    = kwargs['output']
      n_supplied -= 1
    if n_supplied > n_expected:
      raise TypeError(f"expected no more than {n_expected} args, "
                      f"but got {n_supplied}")

    if n_supplied == 0:
      raise TypeError("expected at least one argument to take the "
                      "derivative with respect to.")

    # Step 1: convert all arguments into a {str : str} dict
    #         mapping from variable names to derivative names
    dvars = {}
    arg_i   = 0
    N_kw    = 0
    for vd in self._ast.vars:
      # case: argument was supplied by name
      if str(vd.name) in kwargs:
        val     = kwargs[str(vd.name)]
        N_kw   += 1
      # case: no more positional arguments available
      elif arg_i >= len(args):
        val     = None
      # case: assume argument was supplied by position
      else:
        val     = args[arg_i]
        arg_i  += 1

      # now check that the value in question makes sense
      if val == None or val == False or val == 0:
        # this case means, "no, I don't want to differentiate w.r.t. this"
        pass
      elif val == True:
        # this case means, "yes, but I have no name for the differential"
        dvars[str(vd.name)] = True
      elif type(val) is str:
        # this case means, "yes, and here is the differential's name"
        dvars[str(vd.name)] = val

    if N_kw != len(kwargs):
      raise TypeError("Some named argument was unrecognized.")

    # Step 2: need to check that the differential names don't clash,
    #         and assign safe names to the unnamed differentials
    names   = { str(sym) : True for sym in self._ast.arg_order }
    # check for clashes on supplied names
    for x,dx in dvars.items():
      if type(dx) is str:
        if dx in names:
          raise TypeError(f"Cannot use differential name '{dx}', "
                          f"as it was already in use")
        names[dx] = True
    # come up with fresh names
    unnamed = { x : True for x,dx in dvars.items() if dx is True }
    for x,_ in unnamed.items():
      dx        = "d"+x
      while dx in names:
        dx      = "d"+dx
      dvars[x]  = dx
      names[dx] = True

    # Step 3: output names
    if output is None:
      out         = "out"
      while out in names:
        out      += "_"
      names[out]  = True
      dout        = "d"+out
      while dout in names:
        dout      = "d"+dout
      names[dout] = True
      output = (out,dout)
    else:
      if ( type(output) is not tuple or len(output) != 2 or
           type(output[0]) is not str or type(output[1]) is not str ):
        raise TypeError("expected 'output' argument to be a pair of strings")
      if output[0] in names:
        raise TypeError(f"Output name '{output[0]}' is already in use")
      names[ output[0] ] = True
      if output[1] in names:
        raise TypeError(f"Output differential name '{output[1]}' "
                        f"is already in use")
      names[ output[1] ] = True

    return dvars, output

  def _TEST_BoundCheck(self):
    bd_sys          = BoundsExtraction(self._ast).system()
    BoundsCheck(bd_sys)

  def _TEST_LetLift(self):
    ast             = LetLift(self._ast).normalized()
    return Function(ast, _do_bound_check=False)

  def _TEST_TupleElimination(self):
    ast             = TupleElimination(self._ast).normalized()
    return Function(ast, _do_bound_check=False)

  def _TEST_PreNormalization(self):
    ast             = self._ast
    ast             = LetLift(self._ast).normalized()
    ast             = TupleElimination(ast).normalized()
    ast             = IndexDownGenUp(ast).normalized()
    return Function(ast, _do_bound_check=False)

  def _TEST_TotalDeriv_Alone(self,*args,**kwargs):
    dvars, output   = self._unpack_deriv_args(*args,**kwargs)
    ast             = TotalDerivative(self._ast, dvars, output).result()
    return Function(ast, _do_bound_check=False)

  def _TEST_TotalDeriv(self,*args,**kwargs):
    normed          = self._TEST_PreNormalization()
    dvars, output   = normed._unpack_deriv_args(*args,**kwargs)
    ast             = TotalDerivative(normed._ast, dvars, output).result()
    return Function(ast, _do_bound_check=False)

  def _TEST_PrintNIR(self):
    normed          = self._TEST_PreNormalization()
    nir             = AST_to_NIR(normed._ast).result()
    return str(nir)

  def _TEST_NIR_Roundtrip_NoSimp(self):
    normed          = self._TEST_PreNormalization()
    nir             = AST_to_NIR(normed._ast,use_simplify=False).result()
    ast             = NIR_to_AST(nir).result()
    TC_Lite(ast)
    return Function(ast, _do_bound_check=False)

  def _TEST_NIR_Roundtrip_YesSimp(self):
    normed          = self._TEST_PreNormalization()
    nir             = AST_to_NIR(normed._ast,use_simplify=True).result()
    ast             = NIR_to_AST(nir).result()
    return Function(ast, _do_bound_check=False)

  def _TEST_NIR_Deriv(self,*args,**kwargs):
    dvars, output   = self._unpack_deriv_args(*args,**kwargs)
    normed          = self._TEST_PreNormalization()
    nir             = AST_to_NIR(normed._ast,use_simplify=True).result()
    nir             = NIR_Deriv(nir, dvars).get_deriv()
    ast             = NIR_to_AST(nir).result()
    return Function(ast, _do_bound_check=False)

  def _TEST_NIR_Adjoint(self,*args,**kwargs):
    dvars, output   = self._unpack_deriv_args(*args,**kwargs)
    normed          = self._TEST_PreNormalization()
    nir             = AST_to_NIR(normed._ast,use_simplify=True).result()
    nir             = NIR_Deriv(nir, dvars).get_adjoint()
    ast             = NIR_to_AST(nir).result()
    print(ast)
    return Function(ast, _do_bound_check=False)

