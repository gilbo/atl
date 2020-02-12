"""
Passes to normalize ASTs
"""

from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from .frontend import AST

from fractions import Fraction

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class LetLift:
  """ Pass to raise all Let-bindings to a top-level block

      Call ``normalized()`` to get result

      Parameters
      ==============
      ast : AST.function
        function to normalize let-bindings within

  """
  def __init__(self, ast):
    stmts, expr   = self.lift(ast.body)

    self._ast = AST.function( ast.name, ast.arg_order, ast.rettype,
                              ast.vars, ast.sizes, ast.relations,
                              AST.Let( stmts, expr, expr.type,
                                       ast.body.srcinfo ),
                              ast.srcinfo )

  def normalized(self):
    """ Get the result of lifting; computed on pass construction.

    Returns
    ==============
    AST.function
    """
    return self._ast

  def lift_exprs(self, es):
    """ See ``lift()`` """
    stmts         = []
    exprs         = []
    for e in es:
      s,exp       = self.lift(e)
      stmts      += s
      exprs.append(exp)
    return stmts, exprs

  def lift(self, e):
    """
    Lifts an expression.  If ``[[e]]`` means lift e, then

    ``[[e]] ==> let <return stmts> in <return expr>``

    is an equational representation of this function's signature.
    The rules for individual cases may be deduced from that formulation.

    Parameters
    ==============
    e : AST.expr
      expression to lift

    Returns
    ==============
    [ AST.assign ], AST.expr
        a list of statements and a return expression to build a new
        top level Let-block from

    """
    eclass = type(e)
    if   eclass is AST.Var or eclass is AST.Const:
      return [],e

    elif eclass is AST.BinOp:
      lbind, lhs    = self.lift(e.lhs)
      rbind, rhs    = self.lift(e.rhs)
      return lbind + rbind, AST.BinOp(e.op, lhs, rhs, e.type, e.srcinfo)

    elif eclass is AST.Tuple or eclass is AST.TensorLit:
      stmts, args   = self.lift_exprs(e.args)
      return stmts, eclass(args, e.type, e.srcinfo)

    elif eclass is AST.Proj:
      stmts, arg    = self.lift(e.arg)
      return stmts, AST.Proj(e.idx, arg, e.type, e.srcinfo)

    elif eclass is AST.Gen or eclass is AST.Sum:
      # index variable name
      i     = e.name
      i_rng = e.range

      # process the body
      stmts, body   = self.lift(e.body)

      # the returning let-statement-block needs to be transformed
      # according to the enclosing Gen or Sum.
      #   e.g. Gen(i, let x = e0  => let x = Gen(i, e0)
      #                    in e1)         in Gen(i, e1.sub( x -> x[i] ))
      # In order to make the pass easier to write, but not necessarily
      # efficient, we'll go ahead and eagerly substitute down the entire
      # block of let-statements to take care of this pattern immediately
      ctxt          = Context()
      new_stmts     = []
      for s in stmts:
        # let binding variable name
        x = s.name

        # make a copy of the index variable for this statement
        # and substitute with it and all preceding variables
        sub_i       = i.copy() # index variable name
        ctxt.set(i, sub_i)
        rhs         = self.subst(ctxt, s.rhs)
        tenstyp     = T.Tensor( i_rng, rhs.type )
        new_rhs     = AST.Gen( sub_i, i_rng, rhs, tenstyp, rhs.srcinfo )

        # now that we have a new rhs, we can build up a new assignment
        new_s       = AST.assign( x, new_rhs.type, new_rhs, s.srcinfo )
        new_stmts.append(new_s)
        # and set up the variable for further substitution
        ctxt.set( x, (x,i) )

      # finally, do substitution on the body
      sub_i         = i.copy()
      ctxt.set(i, sub_i)
      body          = self.subst(ctxt, body)
      bodytyp       = body.type
      if eclass is AST.Gen: bodytyp = T.Tensor( i_rng, bodytyp )
      new_body      = eclass(sub_i, i_rng, body, bodytyp, e.srcinfo)

      return new_stmts, new_body

    elif eclass is AST.Access:
      stmts, base   = self.lift(e.base)
      idx           = [ self.lift_index(i) for i in e.idx ]
      return stmts, AST.Access( base, idx, e.type, e.srcinfo )

    elif eclass is AST.BuiltIn:
      stmts, args   = self.lift_exprs(e.args)
      return stmts, AST.BuiltIn( e.f, args, e.type, e.srcinfo )

    elif eclass is AST.Indicate:
      pred          = self.lift_pred(e.pred)
      stmts, body   = self.lift(e.body)
      wrap_stmts    = stmts#[]
      #for s in stmts:
      #  rhs         = AST.Indicate(pred, s.rhs, s.type, s.srcinfo)
      #  wrap_stmts.append(AST.assign( s.name, s.type, rhs, s.srcinfo ))
      return wrap_stmts, AST.Indicate(pred, body, e.type, e.srcinfo)

    elif eclass is AST.Let:
      stmts         = []
      for s in e.stmts:
        block, rhs  = self.lift(s.rhs)
        # accumulate any percolated let-bindings
        stmts      += block
        # also this particular statement
        stmts.append( AST.assign( s.name, s.type, rhs, s.srcinfo ) )
      # and accumulate any statements from the return expression
      block, ret    = self.lift(e.ret)
      stmts        += block

      return stmts, ret

  def lift_index(self, e):
    return e

  def lift_pred(self, p):
    return p

  def subst(self, env, e):
    """
    perform substitutions from ``env`` in ``e`` eagerly.

    Parameters
    ==============
    env : Context
      substitutions to perform
    e : AST.expr
      expression to substitute inside of

    Returns
    ==============
    AST.expr
      The fully substituted expression
    """
    eclass = type(e)
    assert eclass is not AST.Let,   "should not be substituting on lets"
    if eclass is AST.Var:
      sub = env.get(e.name)
      if sub is None: return e
      else:
        idx = env.get(sub[1])
        acc = AST.Access( e, [AST.IdxVar(idx,e.srcinfo)], e.type, e.srcinfo )
        return acc

    elif eclass is AST.Const or eclass is AST.IdxConst:
      return e

    elif eclass is AST.BinOp:
      lhs   = self.subst(env, e.lhs)
      rhs   = self.subst(env, e.rhs)
      return AST.BinOp(e.op, lhs, rhs, e.type, e.srcinfo)

    elif eclass is AST.Tuple or eclass is AST.TensorLit:
      args  = [ self.subst(env, a) for a in e.args ]
      return eclass( args, e.type, e.srcinfo )

    elif eclass is AST.Proj:
      return AST.Proj( e.idx, self.subst(env, e.arg), e.type, e.srcinfo )

    elif eclass is AST.Gen or eclass is AST.Sum:
      assert env.get(e.name) is None, "expected safe symbol distinctions"
      body  = self.subst(env, e.body)
      return eclass( e.name, e.range, body, e.type, e.srcinfo )

    elif eclass is AST.Access:
      base  = self.subst(env, e.base)
      idx   = [ self.subst(env, i) for i in e.idx ]
      return AST.Access( base, idx, e.type, e.srcinfo )

    elif eclass is AST.BuiltIn:
      args  = [ self.subst(env, a) for a in e.args ]
      return AST.BuiltIn( e.f, args, e.type, e.srcinfo )

    elif eclass is AST.Indicate:
      pred  = self.subst(env, e.pred)
      body  = self.subst(env, e.body)
      return AST.Indicate( pred, body, e.type, e.srcinfo )

    elif eclass is AST.IdxVar or eclass is AST.IdxSize:
      new_i = env.get(e.name)
      return e if new_i is None else eclass( new_i, e.srcinfo )

    elif eclass is AST.IdxScale:
      idx   = self.subst(env, e.idx)
      return AST.IdxScale( e.coeff, idx, e.srcinfo )

    elif eclass is AST.Cmp:
      lhs   = self.subst(env, e.lhs)
      rhs   = self.subst(env, e.rhs)
      return AST.Cmp( e.op, lhs, rhs, e.srcinfo )

    elif eclass is AST.Relation:
      args  = [ self.subst(env, a) for a in e.args ]
      return AST.Relation( e.name, args, e.srcinfo )

    elif ( eclass is AST.IdxAdd or eclass is AST.IdxSub or
           eclass is AST.Conj   or eclass is AST.Disj ):
      lhs   = self.subst(env, e.lhs)
      rhs   = self.subst(env, e.rhs)
      return eclass(lhs, rhs, e.srcinfo)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class TupleElimination:
  """ Pass to push all tuple-projection/formation to the input and output

      Code growth is best avoided by first applying the let-lift pass.

      Parameters
      ==============
      ast : AST.function
        function to push tuples out of

  """
  def __init__(self, ast):
    self._ctxt    = Context()

    # bind the input variables in a sealed way
    for vd in ast.vars:
      newvar = AST.Var( vd.name, vd.type, null_srcinfo() )
      self._set_var(vd.name, TupleElimination._Sealed(newvar, input=True))

    # If the body has a tuple return type and is not a Let, we need
    # to wrap it in a Let to ensure correct processing
    pre_body      = ast.body
    if ast.rettype.has_tuples() and type(pre_body) is not AST.Let:
      pre_body    = AST.Let( [], pre_body, pre_body.type, pre_body.srcinfo )
    body          = self.elim(pre_body)
    assert body.type == ast.rettype

    self._ast = AST.function( ast.name, ast.arg_order, ast.rettype,
                              ast.vars, ast.sizes, ast.relations,
                              body, ast.srcinfo )

  class _Sealed:
    """ Helper Values to bind a name to """
    def __init__(self, val, input=False):
      self.input  = input
      self.val    = val

  class _Tuple:
    """ Helper Values to bind a name to """
    def __init__(self, val):
      self.val    = val
    def proj(self, projstk, srcinfo):
      val         = self.val
      for i in reversed(projstk):
        val       = val.args[i]
      # now repackage the remaining value...
      def repack(v):
        if type(v) is AST.Var:
          return AST.Var(v.name, v.type, srcinfo)
        elif type(v) is AST.Tuple:
          args    = [ repack(a) for a in v.args ]
          return AST.Tuple(args, v.type, srcinfo)
      return repack(val)

  def _get_var(self, name):
    return self._ctxt.get(name)
  def _set_var(self, name, subst):
    self._ctxt.set(name, subst)
    return subst

  def normalized(self):
    """ Get the result of tuple elimination; computed on pass construction.

    Returns
    ==============
    AST.function
    """
    return self._ast

  def elim(self, e, projstk = []):
    """
    eliminates tuples from an expression.  If ``[[e]]`` means elim e, then

    ``[[e]].<projstk> ==> <return expr>``

    is an equational representation of this function's signature.
    The rules for individual cases may be deduced from that formulation.

    Parameters
    ==============
    e : AST.expr
      expression to flatten
    projstk : [ int ]
      list of projections to apply; apply last entry first
    outpairs : 

    Returns
    ==============
    AST.expr
        the expression with projections applied, pushed down and pairs lifted

    """
    eclass = type(e)
    if   eclass is AST.Var:
      sub = self._ctxt.get(e.name)

      # if the substitution is of a sealed variable, we want to
      # make sure we replace the variable as is
      if type(sub) is TupleElimination._Sealed:
        # if this is not an input variable, then there should be no
        # projection stack left
        if not sub.input: assert len(projstk) == 0
        node = AST.Var(sub.val.name, sub.val.type, e.srcinfo)
        # if there is a projection stack, we need to dump it on the
        # input variable
        for i in reversed(projstk):
          node = AST.Proj(i, node, node.type.types[i], e.srcinfo)
        return node

      # otherwise the substitution is of a tuple, which we need to
      # deconstruct
      elif type(sub) is TupleElimination._Tuple:
        return sub.proj(projstk, e.srcinfo)

      else: assert False, f"unrecognized lookup: {type(sub)}"

    elif eclass is AST.Const:
      assert len(projstk) == 0
      return e

    elif eclass is AST.BinOp:
      # additions may be between any two equivalent types,
      # but otherwise the types must be scalars
      if e.op != '+': assert len(projstk) == 0
      lhs           = self.elim(e.lhs, projstk.copy())
      rhs           = self.elim(e.rhs, projstk)

      # we must assume that the projstk may have changed the type
      # of this expression
      return AST.BinOp(e.op, lhs, rhs, lhs.type, e.srcinfo)

    elif eclass is AST.Tuple:
      # whenever we encounter a tuple, we ought to have a
      # projection to eliminate it with on hand.
      assert len(projstk) > 0
      # the exception to the above rule is returning tuples at the
      # top-level of the function.  We push that logic into the AST.Let case
      proj_i        = projstk.pop()
      return self.elim(e.args[proj_i], projstk)

    elif eclass is AST.Proj:
      # accumulate projections at the end of the stack
      # these accumulated projections get preference as we apply
      projstk.append(e.idx)
      return self.elim(e.arg, projstk)

    elif eclass is AST.TensorLit:
      # projections are copied into each expression of the
      # tensor literal
      args          = [ self.elim(a, projstk.copy()) for a in e.args ]

      # The type of the result may be different due to
      # projections eliminating tuples in the type
      typ           = T.Tensor(len(args), args[0].type)
      return AST.TensorLit(args, typ, e.srcinfo)

    elif eclass is AST.Gen or eclass is AST.Sum:
      # need to copy variable in case this expression is being
      # duplicated somewhere further up in the AST
      self._ctxt.push()
      i             = self._set_var(e.name, e.name.copy())
      body          = self.elim(e.body, projstk)
      self._ctxt.pop()

      # Patch up the type of the result
      typ           = body.type
      if eclass is AST.Gen: typ = T.Tensor(e.range, typ)
      return eclass(i, e.range, body, typ, e.srcinfo)

    elif eclass is AST.Access:
      base          = self.elim(e.base, projstk)
      idx           = [ self.elim_index(i) for i in e.idx ]
      # need to unroll type len(idx) times
      typ           = base.type
      for _ in idx: # do len(idx) times...
        assert type(typ) is T.Tensor
        typ         = typ.type
      return AST.Access(base, idx, typ, e.srcinfo)

    elif eclass is AST.BuiltIn:
      assert len(projstk) == 0, "did not expect built-ins to return tuples"
      for a in e.args:
        assert not a.type.has_tuples(), "builtin had tuple arguments"
      args          = [ self.elim(a) for a in e.args ]
      return AST.BuiltIn(e.f, args, e.type, e.srcinfo)

    elif eclass is AST.Indicate:
      pred          = self.elim_pred(e.pred)
      body          = self.elim(e.body, projstk)
      return AST.Indicate(pred, body, body.type, e.srcinfo)

    elif eclass is AST.Let:
      # we will recursively expand a statement like
      #     let x : ((T0,T1),T2) = e0
      # into
      #     let x_00 : T0 = [[e0]].0.0
      #     let x_01 : T1 = [[e0]].0.1
      #     let x_1  : T2 = [[e0]].1
      # if x does not have an SoA type, we will perform that
      # type conversion.
      # this process will also build up a term for later substitution
      #   x -> ((x_00,x_01),x_1)
      def soa_unpack( rhs, nm, typ, projstk=[] ):
        # What is the right order to assemble this projstk given that
        # typ is recursively unpacked?
        # Consider a nested tuple type.  The outermost tuple type is
        # the first to be deconstructed/projected; so it needs to occur
        # *last* in the projstk.  Therefore as we descend the type, we
        # need to add new projection indices onto the front of the projstk
        if type(typ) is T.Tuple:
          args    = []
          for i,subtyp in enumerate(typ.types):
            projstk.insert(0,i)
            a     = soa_unpack(rhs,f"{nm}{i}",subtyp,projstk)
            assert a.type == subtyp
            args.append(a)
            projstk.pop(0)
          return AST.Tuple(args, typ, null_srcinfo())
        else:
          # create a copy of the right-hand-side and eliminate
          # tuples on it consistent with this particular branch of
          # the tuple type
          rhs     = self.elim(rhs, projstk.copy())
          assert rhs.type == typ
          # here is the new statement
          x       = Sym(nm)
          stmts.append( AST.assign(x,typ,rhs,stmt.srcinfo) )
          # we also need to "seal" this variable to prevent recursive
          # substitution
          x_var   = AST.Var(x, typ, null_srcinfo())
          self._set_var(x, TupleElimination._Sealed(x_var))
          return x_var

      # we will accumulate all newly generated statements into
      # this one block regardless of their source
      stmts         = []
      self._ctxt.push()
      for stmt in e.stmts:
        # important to not modify name unless necessary
        nm        = str(stmt.name)
        if stmt.type.has_tuples(): nm += '_'
        rsub      = soa_unpack( stmt.rhs, nm, stmt.type.SoA_transform() )

        # bind the SoA-transformed variable now
        self._set_var(stmt.name, TupleElimination._Tuple(rsub))

      # now, when we process the return expression, we may need to
      # also deconstruct it according to its return type...
      rettyp      = e.type.SoA_transform()
      # however, unlike above we have an external projstk coming in.
      # Therefore, we only want to ``soa_unpack`` the still unprojected part
      for i in reversed(projstk):
        rettyp    = rettyp.types[i]

      # often, there won't be any tuples; in that case avoid
      # introducing additional, unnecessary assignments
      if not rettyp.has_tuples():
        ret       = self.elim(e.ret, projstk)
      else:
        ret       = soa_unpack( e.ret, '_', rettyp, projstk )
        # patch up the srcinfo on this return tuple
        ret       = TupleElimination._Tuple(ret).proj([],e.ret.srcinfo)
      self._ctxt.pop()

      return AST.Let( stmts, ret, ret.type, e.srcinfo )

    else: assert False, "unrecognized elim AST case"

  def elim_index(self, i):
    """ See ``elim()`` """
    iclass = type(i)
    if   iclass is AST.IdxConst or iclass is AST.IdxSize:
      return i
    elif iclass is AST.IdxVar:
      return AST.IdxVar( self._get_var(i.name), i.srcinfo )
    elif iclass is AST.IdxAdd or iclass is AST.IdxSub:
      lhs         = self.elim_index(i.lhs)
      rhs         = self.elim_index(i.rhs)
      return iclass( lhs, rhs, i.srcinfo )
    elif iclass is AST.IdxScale:
      idx         = self.elim_index(i.idx)
      return AST.IdxScale( i.coeff, idx, i.srcinfo )
    else: assert False, "unrecognized elim AST case"

  def elim_pred(self, p):
    """ See ``elim()`` """
    pclass = type(p)
    if   pclass is AST.Cmp:
      lhs         = self.elim_index(p.lhs)
      rhs         = self.elim_index(p.rhs)
      return AST.Cmp( p.op, lhs, rhs, p.srcinfo )
    elif pclass is AST.Relation:
      args        = [ self.elim_index(i) for i in p.args ]
      return AST.Relation( p.name, args, p.srcinfo )
    elif pclass is AST.Conj or pclass is AST.Disj:
      lhs         = self.elim_pred(p.lhs)
      rhs         = self.elim_pred(p.rhs)
      return pclass( lhs, rhs, p.srcinfo )
    else: assert False, "unrecognized elim AST case"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class IndexDownGenUp:
  """ Pass to normalize tensor generation and indexing

      Assumes that Let Lifting and Tuple Elimination have been run.
      This pass will move all tensor Generators to the outermost position
      of the right-hand sides of assignments, and will move all Accesses
      to the innermost position, directly wrapping variables.

      Call ``normalized()`` to get result

      Parameters
      ==============
      ast : AST.function
        function to normalize let-bindings within

  """
  def __init__(self, ast):

    self._ctxt  = Context()
    self._in_output = False

    if type(ast.body) is AST.Let:
      stmts     = []
      for s in ast.body.stmts:
        gens, r = self.downup(s.rhs)
        rhs     = self.wrap_gens(gens, r)
        stmts.append( AST.assign(s.name, rhs.type, rhs, s.srcinfo) )
      self._in_output = True
      gens, e   = self.downup(ast.body.ret)
      self._in_output = False
      ret       = self.wrap_gens(gens, e)
      body      = AST.Let(stmts, ret, ret.type, ast.body.srcinfo)
    else:
      self._in_output = True
      gens, e   = self.downup(ast.body.ret)
      self._in_output = False
      body      = self.wrap_gens(gens, e)

    self._ast   = AST.function( ast.name, ast.arg_order, ast.rettype,
                                ast.vars, ast.sizes, ast.relations,
                                body, ast.srcinfo )

  def _get_var(self, name):
    return self._ctxt.get(name)
  def _set_var(self, name, subst):
    self._ctxt.set(name, subst)
    return subst

  def wrap_gens(self, gens, e):
    for i,rng in reversed(gens):
      typ       = T.Tensor(rng, e.type)
      e         = AST.Gen(i,rng,e,typ,e.srcinfo)
    return e
  def wrap_access(self, e, idxstk):
    if len(idxstk) > 0:
      typ       = e.type
      for _ in idxstk:
        typ     = typ.type
      acc       = AST.Access( e, idxstk.copy(), typ, e.srcinfo )
    else:
      acc       = e

    # hack to ensure that tensors on RHS get fully indexed
    if not self._in_output and type(acc.type) is T.Tensor:
      shape     = acc.type.shape()
      igens     = [ (Sym(f"_{i}"),r) for i,r in enumerate(shape) ]
      iexprs    = [ AST.IdxVar(i, null_srcinfo()) for i,r in igens ]
      acc       = AST.Access( acc, iexprs, T.num, acc.srcinfo )
      acc       = self.wrap_gens(igens, acc)

    return acc

  def normalized(self):
    """
    Get the result of lifting; computed on pass construction.

    Returns
    ==============
    AST.function
    """
    return self._ast

  def downup(self, e, idxstk=[]):
    """
    normalizes an expression.  If ``[[e]]`` means downup e, then

    ``[[e]][<idxstk>] ==> <Gen bindings> <expr>``

    is an equational representation of this function's signature.
    The rules for individual cases may be deduced from that formulation.

    Parameters
    ==============
    e : AST.expr
      expression to normalize
    idxstk : [ AST.index ]
      list of expressions to index by; unlike projstk,
      index with entry [0] first

    Returns
    ==============
    [ (Sym, range) ], AST.expr
        a list of Gen bindings and an expression to build a final
        expression from.  Bindings are applied with the last first/innermost

    """
    eclass = type(e)
    assert eclass is not AST.Let, "nested Lets should be eliminated"
    if   eclass is AST.Var:
      return [], self.wrap_access( e, idxstk.copy() )

    elif eclass is AST.Const:
      assert len(idxstk) == 0
      return [], e

    elif eclass is AST.BinOp:
      if e.op != '+': assert len(idxstk) == 0
      # effective tensor dimension...
      eff_dim   = ( 0 if type(e.type) is not T.Tensor
                      else len(e.type.shape()) - len(idxstk) )
      assert eff_dim >= 0

      # the simplest case is that this is effectively a scalar op
      if eff_dim == 0:
        lgen, lhs     = self.downup(e.lhs, idxstk.copy())
        rgen, rhs     = self.downup(e.rhs, idxstk)
        assert len(lgen) == 0
        assert len(rgen) == 0
        return [], AST.BinOp(e.op, lhs, rhs, lhs.type, e.srcinfo)

      # the more complex case is addition, handling Gen bindings
      else:
        lgen, lhs     = self.downup(e.lhs, idxstk.copy())
        # construct indexing corresponding to lgen
        for i,rng in lgen:
          idxstk.insert(0, AST.IdxVar(i, null_srcinfo()))
        # and then process the right-hand side with these additional
        # indexings to ensure that no generators are returned.
        # Doing this accomplishes substitution for the new merged
        # generator variables in lgen
        rgen, rhs     = self.downup(e.rhs, idxstk)
        assert len(rgen) == 0
        return lgen, AST.BinOp('+', lhs, rhs, lhs.type, e.srcinfo)

    elif eclass is AST.Tuple:
      assert len(idxstk) == 0, "tuples should be on the outside"
      args            = []
      typs            = []
      for a in e.args:
        self._in_output = True
        gens, ea      = self.downup(a)
        self._in_output = False
        ea            = self.wrap_gens(gens, ea)
        args.append( ea )
        typs.append( ea.type )
      typ = T.Tuple( e.type.labels, typs )
      return [], AST.Tuple( args, typ, e.srcinfo )

    elif eclass is AST.Proj:
      gens, arg       = self.downup(e.arg)
      assert len(gens) == 0, "projections should be on the inside"
      return [], self.wrap_access( AST.Proj( e.idx, arg,
                                             e.type, e.srcinfo ), idxstk )

    elif eclass is AST.TensorLit:
      # Sometimes literals will compose scalar expressions.  In that
      # special case, let's preserve the Tensor Literal structure
      if False: #len(idxstk) == 0 and e.args[0].type is T.num:
        args          = []
        for a in e.args:
          gens, arg   = self.downup(a)
          assert len(gens) == 0
          args.append(arg)
        return [], AST.TensorLit( args, e.type, e.srcinfo )

      # otherwise, we'll expand the tensor literal using the identity
      #   { e0, e1 }  ==>  Gen[i:2] [i=0]*e0 + [i=1]*e1
      # and then re-direct into the other cases to handle
      else:
        gen_i         = Sym('_i')
        rng           = len(e.args)
        body          = None
        for i,a in enumerate(e.args):
          pred        = AST.Cmp('==', AST.IdxVar(gen_i, a.srcinfo),
                                      AST.IdxConst(i, a.srcinfo),
                                      a.srcinfo)
          term        = AST.Indicate( pred, a, a.type, a.srcinfo )
          if body is None:
            body      = term
          else:
            body      = AST.BinOp( '+', body, term, term.type, e.srcinfo )
        # wrap in a Gen to complete
        node          = AST.Gen( gen_i, rng, body,
                                 T.Tensor(rng, body.type), e.srcinfo )
        return self.downup(node, idxstk)

    elif eclass is AST.Gen:
      if len(idxstk) > 0:
        # then eliminate this generator using the expression to index with
        idx_expr      = idxstk.pop(0)
        self._set_var(e.name, idx_expr)
        return self.downup( e.body, idxstk )
      else:
        # otherwise, just process and return back up with an additional
        # generator on the return stack
        self._set_var( e.name, AST.IdxVar(e.name,null_srcinfo()) )
        gens, body    = self.downup(e.body)
        gens.insert(0, (e.name,e.range) )
        return gens, body

    elif eclass is AST.Sum:
      self._set_var( e.name, AST.IdxVar(e.name,null_srcinfo()) )
      gens, body      = self.downup(e.body, idxstk)
      body            = AST.Sum(e.name, e.range, body, body.type, e.srcinfo)
      return gens, body

    elif eclass is AST.Access:
      idx             = [ self.downup_index(i) for i in e.idx ]
      return self.downup(e.base, idx + idxstk)

    elif eclass is AST.BuiltIn:
      assert len(idxstk) == 0, "did not expect built-ins to return tensors"
      args            = []
      for a in e.args:
        assert a.type == T.num, "builtin had non-scalar arguments"
        gens, arg     = self.downup(a)
        assert len(gens) == 0
        args.append(arg)
      return [], AST.BuiltIn(e.f, args, e.type, e.srcinfo)

    elif eclass is AST.Indicate:
      pred            = self.downup_pred(e.pred)
      gens, body      = self.downup(e.body, idxstk)
      return gens, AST.Indicate(pred, body, body.type, e.srcinfo)

  def downup_index(self, i):
    """ See ``downup()`` """
    iclass = type(i)
    if   iclass is AST.IdxConst or iclass is AST.IdxSize:
      return i
    elif iclass is AST.IdxVar:
      idx             = self._get_var(i.name)
      assert isinstance(idx, AST.index)
      if type(idx) is AST.IdxVar and idx.name == i.name:
        idx           = AST.IdxVar( i.name, i.srcinfo )
      return idx
    elif iclass is AST.IdxAdd or iclass is AST.IdxSub:
      lhs             = self.downup_index(i.lhs)
      rhs             = self.downup_index(i.rhs)
      return iclass( lhs, rhs, i.srcinfo )
    elif iclass is AST.IdxScale:
      idx             = self.downup_index(i.idx)
      return AST.IdxScale( i.coeff, idx, i.srcinfo )

  def downup_pred(self, p):
    """ See ``downup()`` """
    pclass = type(p)
    if   pclass is AST.Cmp:
      lhs             = self.downup_index(p.lhs)
      rhs             = self.downup_index(p.rhs)
      return AST.Cmp( p.op, lhs, rhs, p.srcinfo )
    elif pclass is AST.Relation:
      args            = [ self.downup_index(a) for a in p.args ]
      return AST.Relation( p.name, args, p.srcinfo )
    elif pclass is AST.Conj or pclass is AST.Disj:
      lhs             = self.downup_pred(p.lhs)
      rhs             = self.downup_pred(p.rhs)
      return pclass( lhs, rhs, p.srcinfo )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #







