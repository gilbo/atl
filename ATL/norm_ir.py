
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from . import builtins as B 

from fractions import Fraction
from functools import lru_cache
from collections import namedtuple

import math

from .frontend import AST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# An IR that is a partial normalization of the ATL ASTs

norm_pred_ops = {
  ">"   : True,
  ">="  : True,
  "=="  : True,
}

# Typed AST
NIR = ADT("""
module NIR {
  function  = ( name?       name,
                sym*        arg_order,
                type        rettype,
                var_decl*   vars,
                size_decl*  sizes,
                rel_decl*   relations,
                tuple_out   output,
                srcinfo     srcinfo )

  var_decl  = ( sym name, type   type,  srcinfo srcinfo )
  size_decl = ( sym name,               srcinfo srcinfo )
  rel_decl  = ( sym name, range* sizes, srcinfo srcinfo )
  tuple_out = Tuple( tuple_out* args )
            | OutLeaf( expr expr )
            attributes( type type, srcinfo srcinfo )

  expr  = Var       ( sym       name,   plabel* proj )
        | Const     ( float     val     )
        | Sum       ( float     offset,
                      expr*     terms   )
        | Contract  ( idx_bind* gen_binds,
                      idx_bind* sum_binds,
                      float     coeff, 
                      pred*     preds,
                      factor*   factors )
        | Pow       ( expr base, fraction power )
        | BuiltIn   ( builtin f, expr* args )
        attributes( type type )

  ivar      = ( int name )
  idx_bind  = ( ivar ivar, range range )
  factor    = ( expr expr, idx_bind* idx, fraction power )

  affine    = ( fraction offset, aterm* terms )
  -- exactly one of ivar and size should be present
  aterm     = ( fraction coeff,  ivar? ivar,  sym? size )

  -- ivar here should always be IVar
  pred  = Alias     ( ivar lvar,  ivar rvar )
        | Cmp       ( pred_op op, affine eq ) -- e.g.  eq > 0
        | Relation  ( sym name,   ivar* args )
        | Conj      ( pred* preds )
        | Disj      ( pred* preds )
        | TrueP     ()
        | FalseP    ()
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'builtin':  lambda x: isinstance(x, B.BuiltIn),
  'pred_op':  lambda x: x in norm_pred_ops,
  'plabel':   lambda x: type(x) is int,
  'srcinfo':  lambda x: type(x) is SrcInfo,
})

ADTmemo(NIR,[
  'Var', 'Const', 'Sum', 'Contract', 'Pow', 'BuiltIn',
  'ivar', 'idx_bind', 'factor', 'affine', 'aterm',
  'Cmp', 'Relation', 'Conj', 'Disj', 'TrueP', 'FalseP',
],{
  'name':     lambda x: x,
  'sym':      lambda x: x,
  'type':     id,
  'range':    lambda x: x,
  'fraction': lambda x: x,
  'builtin':  id,
  'pred_op':  lambda x: x,
  'plabel':   lambda x: x,
})

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class NIR_Stmts:
  def __init__(self, nir):
    if isinstance(nir,NIR.expr):
      exprs       = [nir]
    elif isinstance(nir,NIR.tuple_out):
      exprs       = []
      def unpack(tup):
        if type(tup) is NIR.OutLeaf:
          exprs.append( tup.expr )
        else:
          for t in tup.args: unpack(t)
    else: assert False, "expected tuple_out or expr"

    # get counts of each occuring expression
    # in the normalized representation
    self._counts  = {}
    for e in exprs:
      self.visit(e)

  def n_uses(self, e):
    return 0 if e not in self._counts else self._counts[e]

  def is_repeated(self, e):
    return e in self._counts and self._counts[e] > 1

  def visit(self,e):
    if e in self._counts:
      self._counts[e] += 1
      return
    else:
      self._counts[e] = 1
      # and recurse
      eclass      = type(e)
      if eclass is NIR.Var or eclass is NIR.Const:
        pass
      elif eclass is NIR.Sum:
        for ee in e.terms:
          self.visit(ee)
      elif eclass is NIR.Contract:
        for f in e.factors:
          self.visit(f.expr)
      elif eclass is NIR.Pow:
        self.visit(e.base)
      elif eclass is NIR.BuiltIn:
        for ee in e.args:
          self.visit(ee)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class NIR_Check:
  def __init__(self, f):
    self._cache_type      = {}

    # bind type data about inputs
    self._var_types       = {}
    self._rel_sizes       = {}
    for vd in f.vars:       self._var_types[vd.name] = vd.type
    for rd in f.relations:  self._rel_sizes[rd.name] = rd.sizes

    def unpack(out):
      if type(out) is NIR.OutLeaf:
        typ = self.check(out.expr)
        assert typ == out.type
      else:
        for a in out.args: unpack(a)
    unpack(f.output)

  def check(self, e):
    if e in self._cache_type:
      return self._cache_type[e]

    eclass      = type(e)
    if eclass is NIR.Var:
      typ       = self._var_types[e.name]
      for i in e.proj:
        typ     = typ.types[i]
      assert type(typ) is not T.Tuple
      assert typ == e.type

    elif eclass is NIR.Const:
      pass

    elif eclass is NIR.Sum:
      for t in e.terms:
        ttyp    = self.check(t)
        assert ttyp == e.type

    elif eclass is NIR.Contract:
      idx_range   = {}
      def add_ivar(i):
        nonlocal idx_range
        if i.ivar in idx_range:
          print("GEN: ", e.gen_binds)
          print("SUM: ", e.sum_binds)
          print("IDX: ")
          for f in e.factors:
            print("     ", f.idx)
          assert False, "repeated ivar name"
        idx_range[i.ivar] = i.range

      for g in e.gen_binds: add_ivar(g)
      for s in e.sum_binds: add_ivar(s)

      # check factors
      for f in e.factors:
        typ     = self.check(f.expr)
        shape   = typ.shape() if typ != T.num else []
        assert tuple(shape) == tuple(i.range for i in f.idx)
        for i in f.idx: add_ivar(i)

      # check predicates...
      for p in e.preds: self.checkp(p, idx_range)

      # check result type
      shape       = e.type.shape() if e.type != T.num else []
      if tuple(shape) != tuple(g.range for g in e.gen_binds):
        print(shape,e.gen_binds)
        print(e)
      assert tuple(shape) == tuple(g.range for g in e.gen_binds)

    elif eclass is NIR.Pow:
      typ       = self.check(e.base)
      assert typ == e.type

    elif eclass is NIR.BuiltIn:
      shape     = e.type.shape() if e.type != T.num else []
      for a in e.args:
        typ     = self.check(a)
        ranges  = typ.shape() if typ != T.num else []
        assert tuple(shape) == tuple(ranges)

    else: assert False, "unexpected case"

    typ = e.type
    self._cache_type[e] = typ
    return typ

  def checkp(self, p, idx_range):
    pclass = type(p)
    if pclass is NIR.TrueP or pclass is NIR.FalseP:
      pass
    elif pclass is NIR.Conj or pclass is NIR.Disj:
      for pp in p.preds: self.checkp(pp, idx_range)
    elif pclass is NIR.Relation:
      sizes     = self._rel_sizes[p.name]
      arg_rng   = [ idx_range[a] for a in p.args ]
      assert tuple(sizes) == tuple(arg_rng)
    elif pclass is NIR.Cmp:
      for t in p.eq.terms:
        if t.ivar:
          assert t.ivar in idx_range
    elif pclass is NIR.Alias:
      assert p.lvar in idx_range
      assert p.rvar in idx_range
      if idx_range[p.lvar] != idx_range[p.rvar]:
        print("WARNING: found alias predicate with mismatched ranges")
        # can't guarantee that this is an error though...
    else: assert False, "unexpected case"



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class _Stmt_Strs:
  def __init__(self, f):
    used_varnames = {}
    for sz in f.sizes:      used_varnames[str(sz.name)] = True
    for vd in f.vars:       used_varnames[str(vd.name)] = True
    for rd in f.relations:  used_varnames[str(rd.name)] = True
    self._used_varnames   = used_varnames
    self._count           = 0

    self._STMT            = NIR_Stmts(f.output)
    self._expr_var        = {}
    self._stmts           = []

    if type(f.output) is NIR.OutLeaf:
      self._output        = self.getstr(f.output.expr)
    else:
      def unpack(out):
        if type(out) is NIR.OutLeaf:
          return self.getstr(out.expr, bind_expr=True)
        else:
          outs            = ", ".join([ unpack(a) for a in out.args ])
          return f"({outs})"
      self._output        = unpack(f.output)

  def __str__(self):
    return "\n".join(self._stmts + [f"return    {self._output}"])

  def get_new_var(self):
    self._count  += 1
    name          = f"v_{self._count}"
    if name in self._used_varnames:
      return self.get_new_var()
    return name

  # precedence for printing:
  #   Sum       10
  #   Contract  20
  #   Pow       30

  def getstr(self, e, prec=0, bind_expr=False):
    if e in self._expr_var:
      return self._expr_var[e]

    # otherwise compute an actual string
    eclass          = type(e)
    bind_contract   = False
    if eclass is NIR.Var:
      estr          = str(e.name)
      if len(e.proj) > 0:
        estr        += "."
        estr        += ".".join([str(p) for p in e.proj])
    elif eclass is NIR.Const:
      estr          = str(e.val)
    elif eclass is NIR.Sum:
      terms         = [str(e.offset)] + [ self.getstr(t,prec=10)
                                          for t in e.terms ]
      estr          = " + ".join(terms)
      estr = f"({estr})" if prec >= 10 else estr
    elif eclass is NIR.Contract:
      # collect various prefixes into a buffer of lines
      lines         = []
      if len(e.gen_binds) > 0:
        gens        = [ f"i{b.ivar.name}:{b.range}" for b in e.gen_binds ]
        lines.append( f"Gen[{', '.join(gens)}]" )
      if len(e.sum_binds) > 0:
        sums        = [ f"i{b.ivar.name}:{b.range}" for b in e.sum_binds ]
        lines.append( f"Sum[{', '.join(sums)}]" )
      if len(e.preds) > 0:
        preds       = wrapjoin([ self.get_pstr(p) for p in e.preds ],
                               col_width=65, delimit=", ")
        lines.append( "[ "+(",\n" + 12*" ").join(preds)+" ]" )

      factors = [ str(e.coeff) ] if e.coeff != 1.0 else []
      for f in e.factors:
        base        = self.getstr(f.expr, prec=30)
        if f.power != 1:
          base     += '^'+( f"({f.power})" if f.power.denominator != 1
                                           else str(f.power) )
        if len(f.idx) > 0:
          idx       = [ f"i{b.ivar.name}:{b.range}" for b in f.idx ]
          base     += f"[{','.join(idx)}]"
        factors.append(base)
      if len(factors) == 0:
        factors     = ["1.0"]
      fac_lines     = wrapjoin(factors, col_width=66, delimit=" * " )
      lines.append( (" *\n" + 11*" ").join(fac_lines) )

      # now determine whether or not to put things onto one or multiple
      # different lines...
      lines         = wrapjoin(lines, col_width=69, delimit=" ")
      if len(lines) > 1:
        bind_contract = True
      estr          = ("\n" + 10*" ").join(lines)
      if prec >= 20:
        bind_contract = True
      #estr          = f"({estr})" if prec >= 20 else estr

    elif eclass is NIR.Pow:
      base          = self.getstr(e.base, prec=30)
      power         = str(e.power)
      if e.power.denominator == 1: power = f"({power})"
      estr          = f"{base}^{power}"
      estr = f"({estr})" if prec >= 30 else estr
    elif eclass is NIR.BuiltIn:
      args          = ", ".join([ self.getstr(a) for a in e.args ])
      estr          = f"{e.f.name()}({args})"
    else: assert False, "unexpected case"


    # if we decide this expression should be bound to a variable
    bind_this       = ( self._STMT.is_repeated(e) or bind_contract or
                        bind_expr or len(estr) > 65 )
    if bind_this:
      varname       = self.get_new_var()
      self._expr_var[e] = varname
      # pad out to 8
      vbuf          = varname+" "
      if len(vbuf) < 8: vbuf += (8-len(vbuf))*" "

      self._stmts.append(f"{vbuf}= {estr}")
      estr          = varname

    return estr

  def get_pstr(self, p, prec=0):
    pclass    = type(p)
    if pclass is NIR.Alias:
      return f"i{p.lvar.name}=i{p.rvar.name}"
    elif pclass is NIR.Cmp:
      return f"{self.get_astr(p.eq)} {p.op} 0"
    elif pclass is NIR.Relation:
      args    = ",".join([ str(i.name) for i in p.args ])
      return f"{p.name}({p.args})"
    elif pclass is NIR.Conj or pclass is NIR.Disj:
      op      = " and " if pclass is NIR.Conj else " or "
      preds   = op.join([ self.get_pstr(pp,10) for pp in p.preds ])
      if len(p.preds) == 0:
        return "And()" if pclass is NIR.Conj else "Or()"
      return f"({preds})" if (len(p.preds) > 1 and prec > 0) else preds
    elif pclass is NIR.TrueP:
      return "true"
    elif pclass is NIR.FalseP:
      return "false"
    else: assert False, "unexpected case"

  def get_astr(self, a):
    offset          = [str(a.offset)] if a.offset != 0 else []
    terms           = [ ( ("" if t.coeff==1 else f"{t.coeff}*") +
                          (f"i{t.ivar.name}" if t.ivar else "") +
                          (str(t.size) if t.size else "") )
                       for t in a.terms ]
    terms           = offset + terms
    return " + ".join(terms) if len(terms) > 0 else "0"





# NEED a way to print out one of these things....
@extclass(NIR.function)
def __str__(f):
  ind     = "\n         "
  colW    = 80-len(ind)
  sstr    = "sizes    "+ind.join(wrapjoin([ str(sz.name) for sz in f.sizes ],
                                 col_width=colW, delimit=", "))
  vstr    = "vars     "+ind.join(wrapjoin([ f"{str(vd.name)}:{vd.type}"
                                            for vd in f.vars ],
                                          col_width=colW, delimit=", "))
  relnms  = [ str(rd.name) + ','.join([str(s) for s in rd.sizes])
              for rd in f.relations ]
  rstr    = "rels     "+ind.join(wrapjoin(relnms, col_width=colW,
                                                  delimit=", "))

  bstr    = _Stmt_Strs(f)
  nmstr   = "" if f.name is None else f.name

  return f"function {nmstr}\n{sstr}\n{vstr}\n{rstr}\n{bstr}"
del __str__


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

from . import nir_simplify
from .nir_convert import AST_to_NIR, NIR_to_AST





