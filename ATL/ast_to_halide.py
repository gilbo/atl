
from .prelude import *

from . import atl_types as T
from .frontend import AST
from .halide_ir import HIR

from fractions import Fraction

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class Compile:
  def __init__(self, ast):
    self._ast         = ast
    self._ctxt        = Context()
    self._ctxt.push()

    self._vars        = []
    self._rdoms       = []
    self._funcs       = []
    self._params      = []
    self._imgs        = []
    self._stmts       = []
    self._outputs     = []

    self._rel_sizes   = {}

    # bind size variables to pipeline parameters
    for sd in ast.sizes:
      sz_param        = HIR.Param( str(sd.name), HIR.i32 )
      self._params.append(sz_param)
      self._ctxt.set(sd.name, sz_param)

    # bind variables to images or pipeline parameters
    for vd in ast.vars:
      def bind_var(typ=vd.type,name=str(vd.name)):
        if typ == T.num:
          # create a parameter
          v_param     = HIR.Param( name, HIR.f64 )
          self._params.append(v_param)
          return v_param
        elif type(typ) is T.Tensor:
          assert not typ.has_tuples()
          # create an image / func
          n_dims      = len(typ.shape())
          v_img       = HIR.Img( name, n_dims, HIR.f64 )
          self._imgs.append(v_img)
          return HIR.ImgFunc(v_img)
        else:
          assert type(typ) is T.Tuple
          labels      = ( typ.labels.names if typ.labels is not None else
                          [ str(i) for i in range(0,len(typ.types)) ] )
          return tuple( bind_var(subtyp, f"{name}_{label}")
                        for label, subtyp in zip(labels, typ.types) )
      self._ctxt.set( vd.name, bind_var() )

    # bind relations to boolean images
    for rd in ast.relations:
      rngs            = [ self._get_range(sz) for sz in rd.sizes ]
      self._rel_sizes[rd.name] = rngs
      n_dims          = len(rd.sizes)
      v_img           = HIR.Img( str(rd.name), n_dims, HIR.u8 )
      self._imgs.append(v_img)
      self._ctxt.set( rd.name, HIR.ImgFunc(v_img) )

    # process all of the statements
    body              = ast.body
    if type(body) is AST.Let:
      for s in body.stmts:
        self._stmts.append( self._compile_stmt(s) )
      body            = body.ret

    # process the output, potentially as a tuple...
    def bind_output(e=body, name='out'):
      if type(e) is AST.Tuple:
        args          = [ bind_output(a, f"{name}_{i}")
                          for i,a in enumerate(e.args) ]
        return args
      elif type(e.type) is T.Tensor:
        assert type(e) is AST.Var
        out_func = self._ctxt.get(e.name)
        self._outputs.append(out_func)
        return out_func
      else:
        assert e.type is T.num
        out_sym       = Sym(name)
        out_stmt      = AST.assign( out_sym, e.type, e, e.srcinfo )
        self._stmts.append( self._compile_stmt(out_stmt) )
        out_func      = self._ctxt.get(out_sym)
        self._outputs.append(out_func)
        return out_func
    bind_output()

    self._pipeline = HIR.Pipeline(
      vars      = self._vars,
      rdoms     = self._rdoms,
      funcs     = self._funcs,
      params    = self._params,
      imgs      = self._imgs,
      stmts     = self._stmts,
      outputs   = self._outputs,
    )
    #print(self._pipeline)

  def __call__(self, vs, sizes, relations, out):

    # we need to pack lists of the following things
    params      = []
    imgs        = []
    outputs     = []

    # unpack the sizes into parameters first
    for sz in sizes:
      params.append(sz)

    # then unpack the variables
    def unpack_var(v):
      if isinstance(v, tuple):
        for subv in v:
          unpack_var(subv)
      elif type(v) is np.ndarray:
        imgs.append(v)
      else:
        assert type(v) is float
        params.append(v)
    for v in vs:
      unpack_var(v)

    # finally unpack the relations
    for R in relations:
      assert type(R) is np.ndarray
      imgs.append(R)

    # unpack the output in case it's a tuple of buffers
    def unpack_out(val=out):
      if isinstance(val, tuple):
        for o in val:
          unpack_out(o)
      else:
        assert type(val) is np.ndarray
        outputs.append(val)
    unpack_out(out)

    assert len(params)  == len(self._pipeline.params)
    assert len(imgs)    == len(self._pipeline.imgs)
    assert len(outputs) == len(self._pipeline.outputs)
    #print('IMGS\n',imgs)

    # invoke the pipeline
    self._pipeline(params, imgs, outputs)

  def _get_range(self,atl_rng):
    if type(atl_rng) is int:
      return HIR.Range( HIR.Const(0,HIR.i32), HIR.Const(atl_rng,HIR.i32) )
    else:
      assert type(atl_rng) is Sym
      sz_p      = self._ctxt.get(atl_rng)
      assert type(sz_p) is HIR.Param
      return HIR.Range( HIR.Const(0,HIR.i32), HIR.Eparam(sz_p) )

  def _new_rdom(self, name, rng):
    rdom        = HIR.RDom(str(name), [rng])
    self._rdoms.append(rdom)
    self._ctxt.set(name, rdom)
    return rdom

  def _compile_stmt(self, stmt):
    # create Func symbol and bind it
    n_dim       = 1 if stmt.type == T.num else len(stmt.type.shape())
    func        = HIR.Func(str(stmt.name), n_dim, HIR.f64)
    self._ctxt.set(stmt.name, func)
    self._funcs.append(func)

    self._ctxt.push()
    # bind argument/func pure-variables...
    ast_rhs     = stmt.rhs
    if stmt.type == T.num:
      arg_vars  = [ HIR.Var('_0_') ]
      self._vars.append(arg_vars[0])
    elif type(ast_rhs) is AST.Var:
      # special case of tensor aliasing
      shape     = stmt.type.shape()
      arg_vars  = [ HIR.Var(f"_{i}_") for i in range(0,len(shape)) ]
      self._vars += arg_vars
      rhs_func  = self._ctxt.get(ast_rhs.name)
      hir_rhs   = HIR.FAccess(rhs_func, [ HIR.Evar(v) for v in arg_vars ])
      ast_rhs   = None
    else:
      arg_vars  = []
      typ       = stmt.type
      assert typ == ast_rhs.type
      while type(ast_rhs) is AST.Gen:
        hvar    = HIR.Var( str(ast_rhs.name) )
        self._vars.append(hvar)
        self._ctxt.set(ast_rhs.name, hvar)
        arg_vars.append(hvar)
        typ     = typ.type
        ast_rhs = ast_rhs.body
      if typ != T.num:
        print('*****\n*\n*\n*\n',ast_rhs)
      assert typ == T.num
    # compile expression
    if ast_rhs is not None:
      hir_rhs   = self._compile_expr(ast_rhs)
    self._ctxt.pop()

    # return the HIR stmt
    return HIR.PureDef( func, arg_vars, hir_rhs )

  def _compile_leaf(self,e):
    #print('comp leaf ', e)
    # extract accesses
    accesses  = []
    while type(e) is AST.Access:
      accesses = [ self._compile_expr(ie) for ie in e.idx ] + accesses
      e       = e.base

    acc_e     = e

    # extract projections
    projs     = []
    while type(e) is AST.Proj:
      projs.append(e.idx)
      e       = e.arg

    assert type(e) is AST.Var, "expected particular nesting..."

    # unpack any tuple present by projecting statically
    #print('post proj unroll : ', e)
    var_tup   = self._ctxt.get(e.name)
    for i in reversed(projs):
      assert isinstance(var_tup, tuple), (f"expected lookup of "
                                          f"{e.name} to be a tuple")
      var_tup = var_tup[i]
    x         = var_tup

    # now certify that we got a function or param, and generate an access
    if type(x) is HIR.Param:
      assert acc_e.type == T.num
      return HIR.Eparam(x)
    else:
      assert isinstance(x, HIR.func)
      # if this is an input func, protect it with clamping
      if type(x) is HIR.ImgFunc:
        if type(acc_e.type) is not T.Tensor:
          print('XXX', acc_e.type)
          print(acc_e)
        assert type(acc_e.type) is T.Tensor
        rngs      = [ self._get_range(sz) for sz in acc_e.type.shape() ]
        m1        = lambda x: HIR.BinOp('-', x, HIR.Const(1,HIR.i32))
        clamp     = lambda x,hi: HIR.Clamp( x,HIR.Const(0,HIR.i32), m1(hi) )
        accesses  = [ clamp( ie, bd.extent )
                      for ie,bd in zip(accesses, rngs) ]
      # if this is a scalar, then plug in a default access...
      if e.type == T.num:
        assert len(accesses) == 0
        accesses  = [ HIR.Const(0,HIR.i32) ]

      return HIR.FAccess( x, accesses )

  def _compile_expr(self,e):
    eclass    = type(e)

    #   Expr Compile   #

    if   eclass is AST.Var or eclass is AST.Proj or eclass is AST.Access:
      return self._compile_leaf(e)

    elif eclass is AST.Const:
      return HIR.Const(e.val, HIR.f64)

    elif eclass is AST.BinOp:
      lhs     = self._compile_expr(e.lhs)
      rhs     = self._compile_expr(e.rhs)
      return HIR.BinOp(e.op, lhs, rhs)

    elif eclass is AST.Tuple:
      assert False, "Should not be trying to compile Tuples to Halide"

    elif eclass is AST.TensorLit:
      assert False, "TODO: handle tensor literals in Halide compilation"

    elif eclass is AST.Gen:
      assert False, (f"AST.Gen compilation should be handled "
                     f"at the statement level")

    elif eclass is AST.Sum:
      N       = self._get_range(e.range)
      self._ctxt.push()
      r       = self._new_rdom(e.name, N)
      body    = self._compile_expr(e.body)
      self._ctxt.pop()

      return HIR.BigSum(r, body)

    elif eclass is AST.BuiltIn:
      args    = [ self._compile_expr(a) for a in e.args ]
      return e.f.halide_compile(*args)

    elif eclass is AST.Indicate:
      guard   = self._compile_expr(e.pred)
      # maybe indicators should be fused to generators etc?
      return HIR.Select( guard, self._compile_expr(e.body),
                                HIR.Const(0.0,HIR.f64) )

    elif eclass is AST.Let:
      assert False, (f"AST.Let compilation should be handled "
                     f"at the top-level")

    #   Index Exec   #

    elif eclass is AST.IdxConst:
      assert e.val.denominator == 1
      return HIR.Const(e.val.numerator, HIR.i32)
    elif eclass is AST.IdxVar:
      x       = self._ctxt.get(e.name)
      if type(x) is HIR.Var:
        return HIR.Evar(x)
      elif type(x) is HIR.RDom:
        return HIR.Erdom(x)
      else:
        assert False, f"unexpected IdxVar type: {type(x)}"
    elif eclass is AST.IdxSize:
      x       = self._ctxt.get(e.name)
      assert type(x) is HIR.Param
      return HIR.Eparam(x)
    elif eclass is AST.IdxAdd or eclass is AST.IdxSub:
      op      = '+' if eclass is AST.IdxAdd else '-'
      return HIR.BinOp( op, self._compile_expr(e.lhs),
                            self._compile_expr(e.rhs) )
    elif eclass is AST.IdxScale:
      assert e.coeff.denominator == 1
      coeff   = HIR.Const(e.coeff.numerator, HIR.i32)
      return HIR.BinOp('*', coeff, self._compile_expr(e.idx))

    #   Pred Exec   #

    elif eclass is AST.Cmp:
      lhs     = self._compile_expr(e.lhs)
      rhs     = self._compile_expr(e.rhs)
      return HIR.BinOp(e.op, lhs, rhs)

    elif eclass is AST.Relation:
      bds     = self._rel_sizes[e.name]
      m1      = lambda x: HIR.BinOp('-', x, HIR.Const(1,HIR.i32))
      clamp   = lambda x,hi: HIR.Clamp( x, HIR.Const(0,HIR.i32), m1(hi) )
      args    = [ clamp( self._compile_expr(ie), bd.extent )
                  for ie,bd in zip(e.args, bds) ]
      Rfunc   = self._ctxt.get( e.name )
      Acc     = HIR.FAccess(Rfunc, args)
      return HIR.BinOp("!=",Acc,HIR.Const(0,HIR.u8))

    elif eclass is AST.Conj or eclass is AST.Disj:
      op      = "and" if eclass is AST.Conj else "or"
      lhs     = self._compile_expr(e.lhs)
      rhs     = self._compile_expr(e.rhs)
      return HIR.BinOp(op, lhs, rhs)
