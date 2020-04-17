
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

  # ----------------------------------------------------------------------- #
  #   Introspection                                                         #
  # ----------------------------------------------------------------------- #

  def name(self):
    return self._ast.name

  def return_type(self):
    return self._ast.rettype

  def arg_list(self):
    """ list of `(name,type)` pairs where `type` is `"size"` if the
        argument is a size and `type` is `"relation"` if the argument is
        a relation
    """
    return [ ( (str(d.name),d.type) if type(d) is AST.var_decl else
               (str(d.name),"size") if type(d) is AST.size_decl else
               (str(d.name),"relation")
             ) for d in self._ast.decls_in_order() ]

  # ----------------------------------------------------------------------- #
  #   Execution                                                             #
  # ----------------------------------------------------------------------- #

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
      assert type(output[0]) == np.float64
      return float(output[0])
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

  # ----------------------------------------------------------------------- #
  #   Derivatives and Transformations                                       #
  # ----------------------------------------------------------------------- #

  def deriv(self,*args,**kwargs):
    return self._nir_deriv_func(*args,**kwargs)

  def adjoint(self,*args,**kwargs):
    return self._nir_adjoint_func(*args,**kwargs)

  def partial_eval(self,*args,**kwargs):
    return self._do_partial_eval(*args,**kwargs)._nir_simplify_func()

  def simplify(self):
    return self._nir_simplify_func()

  def proj(self, label):
    rtyp  = self.return_type()
    if type(rtyp) != T.Tuple:
      raise TypeError(f"can only project functions with Tuple outputs")
    idx   = -1
    if type(label) is int:
      if label < 0 or label >= len(rtyp.types):
        raise TypeError(f"projection index {label} is out of bounds")
      idx = label
    elif type(label) is str:
      if rtyp.labels is None:
        raise TypeError(f"cannot project an unlabeled tuple type "
                        f"{rtyp} with strings")
      for i,nm in enumerate(rtyp.labels.names):
        if nm == label:
          idx = i
          break
      if idx == -1:
        raise TypeError(f"cannot find label '{label}' in tuple type "
                        f"{rtyp}")
    else: raise TypeError("Expected int or str argument")

    if not hasattr(self, '_proj_cache'):
      self._proj_cache = dict()
    if idx not in self._proj_cache:
      body = self._ast.body
      body = AST.Proj(idx, self._ast.body, rtyp.types[idx], body.srcinfo)
      ast = AST.function(self._ast.name,
                         self._ast.arg_order,
                         rtyp.types[idx],
                         self._ast.vars,
                         self._ast.sizes,
                         self._ast.relations,
                         body,
                         self._ast.srcinfo)
      self._proj_cache[idx] = Function(ast, _do_bound_check=False)
    return self._proj_cache[idx]

  def grad(self,*args,**kwargs):
    if self.return_type() != T.num:
      raise TypeError(f"cannot take the gradient of a function with a "
                      f"non-scalar return type: {self.return_type()}")
    adj     = self.adjoint(*args,**kwargs)
    anames  = [ a[0] for a in adj.arg_list() if type(a[1]) != str ]
    lastnm  = anames[-1]
    return adj.partial_eval(**({ lastnm : 1.0 }))

  def _prenorm_ast(self):
    if not hasattr(self, '_prenorm_ast_cached'):
      ast           = self._ast
      ast           = LetLift(self._ast).normalized()
      ast           = TupleElimination(ast).normalized()
      ast           = IndexDownGenUp(ast).normalized()
      self._prenorm_ast_cached = ast
    return self._prenorm_ast_cached

  def _unpack_deriv_args_key(self,*args,**kwargs):
    dvars, output   = self._unpack_deriv_args(*args,**kwargs)
    dvkey           = [(x,dx) for x,dx in dvars.items()]
    dvkey.sort()
    key             = (tuple(dvkey), output)
    return dvars, output, key

  def _nir_deriv_func(self,*args,**kwargs):
    dvars, output, key  = self._unpack_deriv_args_key(*args,**kwargs)
    if not hasattr(self, '_nir_deriv_func_cache'):
      self._nir_deriv_func_cache = dict()
    if not key in self._nir_deriv_func_cache:
      ast           = self._prenorm_ast()
      nir           = AST_to_NIR(ast,use_simplify=True).result()
      nir           = NIR_Deriv(nir, dvars, output).get_deriv()
      ast           = NIR_to_AST(nir).result()
      self._nir_deriv_func_cache[key] = Function(ast, _do_bound_check=False)
    return self._nir_deriv_func_cache[key]

  def _nir_adjoint_func(self,*args,**kwargs):
    dvars, output, key  = self._unpack_deriv_args_key(*args,**kwargs)
    if not hasattr(self, '_nir_adjoint_func_cache'):
      self._nir_adjoint_func_cache = dict()
    if not key in self._nir_adjoint_func_cache:
      ast           = self._prenorm_ast()
      nir           = AST_to_NIR(ast,use_simplify=True).result()
      nir           = NIR_Deriv(nir, dvars, output).get_adjoint()
      ast           = NIR_to_AST(nir).result()
      self._nir_adjoint_func_cache[key] = Function(ast, _do_bound_check=False)
    return self._nir_adjoint_func_cache[key]

  def _nir_simplify_func(self):
    if not hasattr(self, '_nir_simplify_func_cache'):
      ast           = self._prenorm_ast()
      nir           = AST_to_NIR(ast,use_simplify=True).result()
      ast           = NIR_to_AST(nir).result()
      self._nir_simplify_func_cache = Function(ast, _do_bound_check=False)
    return self._nir_simplify_func_cache

  def _do_partial_eval(self, *args, **kwargs):
    # Step 0: check that not too many arguments were supplied.
    n_supplied  = len(args) + len(kwargs)
    n_expected  = len(self._ast.vars)
    if n_supplied > n_expected:
      raise TypeError(f"expected no more than {n_expected} args, "
                      f"but got {n_supplied}")

    if n_supplied == 0:
      raise TypeError("expected at least one argument to partially evaluate.")

    # Step 1: convert all supplied arguments into a {sym : val} dict
    #         mapping from variable names to values
    argmap      = dict()
    arg_i   = 0
    N_kw    = 0
    for vd in self._ast.vars:
      val = None
      # case: argument was supplied by name
      if str(vd.name) in kwargs:
        val     = kwargs[str(vd.name)]
        N_kw   += 1
      # case: was argument was supplied by position?
      elif arg_i < len(args):
        val     = args[arg_i]
        arg_i  += 1

      # check that value and type make sense
      if val is not None:
        if not vd.type.is_const_size():
          raise TypeError(f"Can only substitute arguments of constant size; "
                          f"'{str(vd.name)}' does not have constant size")
        else:
          argcheck_python_value(Context(), vd.type, val, vd.name)
          argmap[vd.name] = val

    if N_kw != len(kwargs):
      raise TypeError("Some named argument was unrecognized.")

    # Step 2: construct new declaration list
    #         and let-bindings for now supplied values
    new_decls   = []
    binds       = []
    for d in self._ast.decls_in_order():
      if type(d) is AST.size_decl or type(d) is AST.rel_decl:
        new_decls.append(d)
      elif type(d) is AST.var_decl:
        # argument that was not supplied for evaluation
        if d.name not in argmap:
          new_decls.append(d)
        # argument that was supplied for evaluation
        else:
          val       = argmap[d.name]
          val_ast   = pyval_to_ast(Context(), d.type, val, d.srcinfo)
          binds.append(AST.assign(d.name, d.type, val_ast, d.srcinfo))
      else: error("impossible case")

    body = AST.Let( binds, self._ast.body,
                    self._ast.body.type, self._ast.srcinfo )

    # Step 3: construct a new function AST
    f_ast = AST.function(self._ast.name,
                         [ d.name for d in new_decls ],
                         self._ast.rettype,
                         [ d for d in new_decls if type(d) is AST.var_decl ],
                         [ d for d in new_decls if type(d) is AST.size_decl ],
                         [ d for d in new_decls if type(d) is AST.rel_decl ],
                         body,
                         self._ast.srcinfo)

    return Function(f_ast, _do_bound_check=False)


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

  # ----------------------------------------------------------------------- #
  #   Testing                                                               #
  # ----------------------------------------------------------------------- #

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
    ast             = self._prenorm_ast()
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
    nir             = NIR_Deriv(nir, dvars, output).get_deriv()
    ast             = NIR_to_AST(nir).result()
    return Function(ast, _do_bound_check=False)

  def _TEST_NIR_Adjoint(self,*args,**kwargs):
    dvars, output   = self._unpack_deriv_args(*args,**kwargs)
    normed          = self._TEST_PreNormalization()
    nir             = AST_to_NIR(normed._ast,use_simplify=True).result()
    nir             = NIR_Deriv(nir, dvars, output).get_adjoint()
    ast             = NIR_to_AST(nir).result()
    #print(ast)
    return Function(ast, _do_bound_check=False)

