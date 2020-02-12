"""
Passes to take derivatives of ASTs
"""

from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from .frontend import AST

from fractions import Fraction

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class TotalDerivative:
  """ Pass to take the derivative of a function.

      If ``f(x : X, y : Y) -> T`` is the function we are differentiating,
      then we must say here which of `x` and `y` we are taking the
      derivative with respect to.  e.g. we can supply the variable mapping
      ``{ 'x' : 'dx' }`` which means we will differentiate w.r.t. ``x``,
      but not ``y``.  In this case, the total derivative ``Df`` of ``f``
      has signature

        `` Df(x : X, y : Y, dx : X) -> (T,T) ``

      Notice that the return type is now a tuple.  The first element of
      the tuple is the original function's output.  The second element
      is the differential output.

      From this form, other desired forms of the derivative can be
      computed by projecting the output to choose which values are
      actually desired.

      Call ``result()`` to get the resulting AST.function

      Parameters
      ==============
      ast : AST.function
        function to take total derivative of
      dvars : { str : str }
        dict mapping strings corresponding to argument variables
        to strings corresponding to new differential argument variables.
  """
  def __init__(self, ast, dvars, outname_pair):
    # set of currently used variable names
    arg_strs    = { str(x) : True for x in ast.arg_order }
    # map: orig_var -> decl
    arg_decls   = { str(vd.name) : vd for vd in ast.vars }

    # Check the following:
    #     *   (_:dx),(_:dy) in dvars ==> dx != dy  (no repeats)
    assert ( len([ dx for x,dx in dvars.items() ]) ==
             len({ dx : True for x,dx in dvars.items() })
           ), "found repeated differential names"
    #     *   (x:dx) in dvars ==> x is a Var arg, and dx is not an arg
    for x,dx in dvars.items():
      assert type(x) is str and type(dx) is str
      assert x in arg_decls, f"did not find '{x}' as a Var arg"
      assert dx not in arg_strs, f"differential '{dx}' is already an arg"

    # Now having checked the mapping, construct a symbolic version of
    # it, as well as updated argument lists
    self._ctxt  = Context()
    new_vars    = ast.vars.copy()
    new_arg_ord = ast.arg_order.copy()
    for xsym in ast.arg_order:
      if str(xsym) in dvars:
        dsym    = Sym(dvars[str(xsym)])
        dtyp    = arg_decls[str(xsym)].type
        srcinfo = arg_decls[str(xsym)].srcinfo

        self._ctxt.set(xsym, dsym)
        new_vars.append( AST.var_decl(dsym, dtyp, srcinfo) )
        new_arg_ord.append( dsym )

    # the new output type for the function is
    out, dout   = outname_pair
    rettype     = T.Tuple(T.labels([out,dout]), [ast.rettype, ast.rettype])

    # compute the derviative of the body...
    d_body      = self.deriv(ast.body)
    # in the common case of a top-level let-binding, we can be clever
    if type(ast.body) is AST.Let:
      ret       = AST.Tuple([ast.body.ret, d_body.ret],
                            rettype, ast.body.ret.srcinfo)
      body      = AST.Let( d_body.stmts, ret, rettype, ast.body.srcinfo )
    else:
      body      = AST.Tuple([ast.body, d_body], rettype, ast.body.srcinfo)

    name        = f"D_{ast.name}" if ast.name else None
    self._ast   = AST.function( name, new_arg_ord, rettype,
                                new_vars, ast.sizes, ast.relations,
                                body, ast.srcinfo )

  def result(self):
    """ Get the total derivative AST; computed on pass construction.

    Returns
    ==============
    AST.function
    """
    return self._ast

  def gen_zero(self, typ, srcinfo):
    if typ is T.num:
      return AST.Const(0.0,typ,srcinfo)
    elif type(typ) is T.Tuple:
      args  = [ self.gen_zero(t,srcinfo) for t in typ.types ]
      return AST.Tuple(args, typ, srcinfo)
    elif type(typ) is T.Tensor:
      body  = self.gen_zero(typ.type,srcinfo)
      return AST.Gen( Sym('_'), typ.range, body, typ, srcinfo )
    else: assert False, "impossible case"

  def deriv(self, e):
    """
    Differentiates an expression.  If ``D[[e]]`` means derivative of e, then

    ``D[[e]] ==> d_e``

    is an equational representation of this function's signature.
    The rules for individual cases may be deduced from that formulation.

    Parameters
    ==============
    e : AST.expr
      expression to derive

    Returns
    ==============
    AST.expr
    """

    eclass = type(e)
    if   eclass is AST.Var:
      d_nm        = self._ctxt.get(e.name)
      if d_nm is None:  return self.gen_zero(e.type, e.srcinfo)
      else:             return AST.Var( d_nm, e.type, e.srcinfo )

    elif eclass is AST.Const:
      return AST.Const( 0.0, T.num, e.srcinfo )

    elif eclass is AST.BinOp:
      dlhs          = self.deriv(e.lhs)
      drhs          = self.deriv(e.rhs)
      if e.op == '+' or e.op == '-':
        return AST.BinOp( e.op, dlhs, drhs, e.type, e.srcinfo )
      elif e.op == '*' or e.op == '/':
        dlr         = AST.BinOp( '*', dlhs, e.rhs, e.type, e.srcinfo )
        ldr         = AST.BinOp( '*', e.lhs, drhs, e.type, e.srcinfo )
        if e.op == '*':
          return AST.BinOp( '+', dlr, ldr, T.num, e.srcinfo )
        else:
          rr        = AST.BinOp( '*', e.rhs, e.rhs, e.type, e.srcinfo )
          numer     = AST.BinOp( '-', dlr, ldr, T.num, e.srcinfo )
          return AST.BinOp( '/', numer, rr, T.num, e.srcinfo )
      else: assert False, f"unrecognized op: {e.op}"

    elif eclass is AST.Tuple or eclass is AST.TensorLit:
      dargs       = [ self.deriv(a) for a in e.args ]
      return eclass( dargs, e.type, e.srcinfo )

    elif eclass is AST.Proj:
      darg        = self.deriv(e.arg)
      return AST.Tuple( e.idx, darg, e.type, e.srcinfo )

    elif eclass is AST.Gen or eclass is AST.Sum:
      dbody       = self.deriv(e.body)
      return eclass( e.name, e.range, dbody, e.type, e.srcinfo )

    elif eclass is AST.Access:
      dbase       = self.deriv(e.base)
      return AST.Access( dbase, e.idx, e.type, e.srcinfo )

    elif eclass is AST.BuiltIn:
      dargs       = [ self.deriv(a) for a in e.args ]
      return e.f.deriv( *(e.args + dargs), srcinfo = e.srcinfo )

    elif eclass is AST.Indicate:
      dbody       = self.deriv(e.body)
      return AST.Indicate( e.pred, dbody, e.type, e.srcinfo )

    elif eclass is AST.Let:
      stmts       = []
      self._ctxt.push()
      for s in e.stmts:
        x, dx     = s.name, Sym('d'+str(s.name))
        self._ctxt.set(x,dx)
        drhs      = self.deriv(s.rhs)
        stmts.append(s)
        stmts.append(AST.assign( dx, s.type, drhs, s.srcinfo ))

      dret        = self.deriv(e.ret)
      self._ctxt.pop()

      return AST.Let( stmts, dret, e.type, e.srcinfo )

    else: assert False, f"unrecognized case {eclass}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #



























