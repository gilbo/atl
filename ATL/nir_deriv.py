
from .prelude import *

from . import atl_types as T

from fractions import Fraction
from collections import namedtuple

import math

from .norm_ir import NIR, NIR_Stmts, nir_mul


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class NIR_Deriv:
  """ Pass to take the derivative or adjoint-derivative of a function
      in normalized-IR form.

      If ``f(x : X, y : Y) -> T`` is the function we are differentiating,
      then we must say here which of `x` and `y` we are taking the
      derivative with respect to.  e.g. we can supply the variable mapping
      ``{ 'x' : 'dx' }`` which means we will differentiate w.r.t. ``x``,
      but not ``y``.  In this case, the derivative ``Df`` of ``f``
      has signature

        `` Df(x : X, y : Y, dx : X) -> (T,T) ``

      and the adjoint-derivative ``D^Tf`` of ``f`` has signature

        `` D^Tf(x : X, y : Y, dout : T) -> (T,X)

      Notice that the return type in both cases is now a pair.
      The first element of the pair is the original function's output. 
      The second element is the differential output, whether that's
      ``dout`` as in the case of the derivative, or ``dx`` as in the case
      of the adjoint-derivative.

      From this form, other desired forms of the derivative can be
      computed by projecting the output to choose which values are
      actually desired.

      Call ``get_deriv()`` to get the derivative NIR.function.
      Call ``get_adjoint()`` to get the adjoint-derivative NIR.function

      Parameters
      ==============
      nir : NIR.function
        function to take (adjoint-)derivative of
      dvars : { str : str }
        dict mapping strings corresponding to argument variables
        to strings corresponding to new differential argument variables.
  """
  def __init__(self, nir, dvars, outname_pair, use_simplify=True):
    self._nir   = nir
    self._use_simplify = use_simplify
    self._outname_pair = outname_pair

    # set of all currently used variable names
    all_strs    = { str(x) : True for x in nir.arg_order }
    # map: orig_var -> decl
    arg_decls   = { str(vd.name) : vd for vd in nir.vars }

    # Check the following:
    #     *   (_:dx),(_:dy) in dvars ==> dx != dy  (no repeats)
    assert ( len([ dx for x,dx in dvars.items() ]) ==
             len({ dx : True for x,dx in dvars.items() })
           ), "found repeated differential names"
    #     *   (x:dx) in dvars ==> x is a Var arg, and dx is not an arg
    for x,dx in dvars.items():
      assert type(x) is str and type(dx) is str
      assert x in arg_decls, f"did not find '{x}' as a Var arg"
      assert dx not in all_strs, f"differential '{dx}' is already an arg"

    #new_vars    = ast.vars.copy()
    #new_arg_ord = ast.arg_order.copy()
    self._ctxt  = Context()
    self._d_decl = []
    for xsym in nir.arg_order:
      if str(xsym) in dvars:
        dsym    = Sym(dvars[str(xsym)])
        dtyp    = arg_decls[str(xsym)].type
        srcinfo = arg_decls[str(xsym)].srcinfo

        vd      = NIR.var_decl(dsym, dtyp, srcinfo)
        self._ctxt.set(xsym, vd)
        self._d_decl.append( vd )

  def get_deriv(self):
    nir         = self._nir

    self._e_deriv   = {}

    arg_order   = nir.arg_order + [ vd.name for vd in self._d_decl ]
    newvars     = nir.vars + self._d_decl
    outnm, doutnm = self._outname_pair
    rettype     = T.Tuple(T.labels([outnm,doutnm]),
                                   [nir.rettype, nir.rettype])

    self._ctxt.push()
    d_out       = self.deriv(nir.output)
    out         = NIR.Tuple( [ nir.output, d_out ],
                             rettype, nir.output.srcinfo )
    self._ctxt.pop()

    name        = f"D_{nir.name}" if nir.name else None
    final       = NIR.function( name, arg_order, rettype,
                                newvars, nir.sizes, nir.relations,
                                out, nir.srcinfo )
    return final

  def deriv(self, e):
    if e in self._e_deriv:
      return self._e_deriv[e]
    eclass        = type(e)
    # tuple_outs
    if eclass is NIR.OutLeaf:
      ederiv      = NIR.OutLeaf( self.deriv(e.expr), e.type, e.srcinfo )

    elif eclass is NIR.Tuple:
      args        = [ self.deriv(a) for a in e.args ]
      ederiv      = NIR.Tuple( args, e.type, e.srcinfo )

    # expressions
    elif eclass is NIR.Var:
      vd          = self._ctxt.get(e.name)
      if vd:
        ederiv    = NIR.Var( vd.name, e.proj, e.type )
      else:
        ederiv    = NIR.Const( 0.0, e.type )

    elif eclass is NIR.Const:
      ederiv      = NIR.Const( 0.0, e.type )

    elif eclass is NIR.Sum:
      ederiv      = NIR.Sum(0.0, [ self.deriv(t) for t in e.terms ], e.type)

    elif eclass is NIR.Contract:
      dfactors    = []
      for f in e.factors:
        df = self.deriv( NIR.Pow(f.expr,f.power,f.expr.type) )
        dfactors.append( NIR.factor(df, f.idx, Fraction(1)) )
      terms       = []
      for i,df in enumerate(dfactors):
        sub_factors = [ f for f in e.factors ]
        sub_factors[i] = df
        terms.append(NIR.Contract( e.gen_binds,
                                   e.sum_binds,
                                   e.coeff,
                                   e.preds,
                                   sub_factors,
                                   e.type ))
      ederiv      = NIR.Sum( 0.0, terms, e.type )

    elif eclass is NIR.Pow:
      if e.power == 0:
        ederiv    = NIR.Const( 0.0, e.type )
      else:
        dbase     = self.deriv(e.base)
        if e.power == 1:
          ederiv  = dbase
        else:
          ederiv  = nir_mul( NIR.Pow( e.base, e.power-1, e.type ),
                             dbase, coeff=float(e.power) )

    elif eclass is NIR.BuiltIn:
      dargs       = [ self.deriv(a) for a in e.args ]
      ederiv      = e.f.nir_deriv( *(e.args + dargs) )
    else: assert False, "unexpected case"

    if self._use_simplify and isinstance(e, NIR.expr):
      ederiv      = ederiv.simplify()
    self._e_deriv[e]  = ederiv
    return ederiv

  def get_adjoint(self):
    nir         = self._nir

    self._stmts = NIR_Stmts(nir.output) # call n_uses() to get out-degree
    self._accum = {}

    outnm, doutnm = self._outname_pair
    doutSym     = Sym(doutnm)
    vd_dout     = NIR.var_decl(doutSym, nir.rettype, nir.srcinfo)
    arg_order   = nir.arg_order + [ doutSym ]
    newvars     = nir.vars + [ vd_dout ]

    self._ctxt.push() # probably unnecessary
    # first, prime all the final accumulator structures
    def create_out_tuple(typ):
      if type(typ) is T.Tuple:
        return tuple( create_out_tuple(subT) for subT in typ.types )
      else:
        return [] # accumulation list
    for vd in self._d_decl:
      self._accum[vd.name] = create_out_tuple(vd.type)

    # process the body of the function
    def unpack(out,proj=[]):
      if type(out) is NIR.Tuple:
        for i,a in enumerate(out.args):
          unpack(a,[i]+proj)
      else:
        assert type(out) is NIR.OutLeaf
        # create an associated differential variable node as input to adjoint
        diffT   = NIR.Var( doutSym, proj, out.type )
        self.adjoint(out.expr, diffT)
    unpack(nir.output)

    # now extract the accumulated outputs
    def assemble_out_tuple(acc,typ,srcinfo):
      if type(typ) is T.Tuple:
        return NIR.Tuple([ assemble_out_tuple(a,t,srcinfo)
                           for a,t in zip(acc,tup.types) ], typ, srcinfo)
      else:
        acc_e   = NIR.Sum(0.0, acc, typ)
        if self._use_simplify:
          acc_e = acc_e.simplify()
        return NIR.OutLeaf( acc_e, typ, srcinfo )
    acc_outs    = [ assemble_out_tuple(self._accum[vd.name],
                                       vd.type, vd.srcinfo)
                    for vd in self._d_decl ]
    # and if there are more than one such output, assemble them into an
    # output tuple
    assert len(acc_outs) > 0
    if len(acc_outs) == 1:
      g_out     = acc_outs[0]
    else:
      labels    = T.labels([ str(vd.name) for vd in self._d_decl ])
      d_rettype = T.Tuple(labels, [ vd.type for vd in self._d_decl ])
      g_out     = NIR.Tuple(acc_outs, d_rettype, nir.srcinfo)

    # now assemble the output pair...
    rettype     = T.Tuple(T.labels([outnm,'grad_out']),
                                   [nir.rettype, g_out.type])
    out         = NIR.Tuple([nir.output, g_out], rettype, nir.srcinfo)
    self._ctxt.pop()

    name        = f"Adj_{nir.name}" if nir.name else None
    final       = NIR.function( name, arg_order, rettype,
                                newvars, nir.sizes, nir.relations,
                                out, nir.srcinfo )
    return final

  def adjoint(self, e, diffT):
    assert e.type == diffT.type

    # collect the incoming differential; should we proceed?
    if e not in self._accum:
      self._accum[e]  = [diffT]
    else:
      self._accum[e].append(diffT)
    n_remain    = self._stmts.n_uses(e) - len(self._accum[e])
    if n_remain > 0:
      return # Wait until we've accumulated all the contributions
    else:
      assert self._stmts.n_uses(e) == 0 or n_remain == 0

    # if we've accumulated multiple things, sum them together...
    if len(self._accum[e]) > 1:
      diffT     = NIR.Sum(0.0, self._accum[e], e.type)

    self._adjoint_helper(e, diffT)

  def _adjoint_helper(self, e, diffT):
    eclass        = type(e)
    if eclass is NIR.Var:
      vd          = self._ctxt.get(e.name)
      if vd:
        assert vd.name in self._accum
        lst       = self._accum[vd.name]
        for i in e.proj:
          lst     = lst[i]
        lst.append(diffT)
      else:
        return # do nothing, i.e. zero

    elif eclass is NIR.Const:
      return # do nothing, i.e. zero

    elif eclass is NIR.Sum:
      # distribute copies of this differential to each term
      for t in e.terms:
        self.adjoint( t, diffT )

    elif eclass is NIR.Contract:
      remap       = Context()
      counter     = 0
      def fresh_binds(bds):
        nonlocal counter
        c         = counter
        counter  += len(bds)
        newbds    = [ NIR.idx_bind(NIR.ivar(i+c), b.range)
                      for i,b in enumerate(bds) ]
        for old,new in zip(bds,newbds):
          remap.set(old.ivar, new.ivar)
        return newbds

      # Create a copy of the factors to represent the differentials we will
      # be propagating down
      dfactors    = [ f for f in e.factors ]
      # then run through the factors and flip each one with the generator...
      for df,f in zip(dfactors, e.factors):
        remap.push()
        gen       = fresh_binds(f.idx)
        subfacs   = [NIR.factor(diffT, fresh_binds(e.gen_binds), Fraction(1))]
        subfacs  += [ NIR.factor(sf.expr, fresh_binds(sf.idx), sf.power)
                      for sf in e.factors if sf != f ]
        sums      = fresh_binds(e.sum_binds)
        preds     = [ p.subst(remap) for p in e.preds ]
        prod      = NIR.Contract( gen, sums, e.coeff,
                                  preds, subfacs, f.expr.type )
        # handle exponent (see NIR.Pow case for reference)
        if df.power == 0:
          continue
        elif df.power == 1:
          self.adjoint(df.expr, prod)
        else:
          diffT   = nir_mul( NIR.Pow(df.expr, df.power-1, df.expr.type),
                             prod, coeff=float(df.power) )
          self.adjoint(df.expr, diffT)
        remap.pop()
        counter   = 0

    elif eclass is NIR.Pow:
      if e.power == 0:
        return # do nothing, i.e. zero
      else:
        if e.power == 1:
          self.adjoint( e.base, diffT )
        else:
          diffT   = nir_mul( NIR.Pow( e.base, e.power-1, e.type ),
                             diffT, coeff=float(e.power) )
          self.adjoint( e.base, diffT )

    elif eclass is NIR.BuiltIn:
      dargs       = e.f.nir_adjoint( diffT, *e.args )
      for a,da in zip(e.args,dargs):
        self.adjoint( a, da )
    else: assert False, "unexpected case"














