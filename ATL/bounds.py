
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from .frontend import AST

from fractions import Fraction
from math import gcd as _gcd
def _lcm(x,y):
  return (x*y)//_gcd(x,y)

# notes on symbols to use
# Note that BOOL is a type and Bool a formula constructor function
#from pysmt.shortcuts import (
#    Symbol, BOOL, INT, REAL,
#    Bool, Int, Real,
#    TRUE, FALSE, And, Or, Not,
#    GT, GE, LE, LT, Equals, NotEquals,
#    Plus, Minus, Times, Div   
#)
import pysmt
from pysmt import shortcuts as SMT

def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers()
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Description of Bounds-Checking Problem

BD = ADT("""
module BD {
  sys     = VarIntro  ( sym   name, sys cont, srcinfo srcinfo )
          | RelIntro  ( sym   name, int n_args,
                                    sys cont, srcinfo srcinfo )
          | Guard     ( pred  pred, sys cont, srcinfo srcinfo )
          | Both      ( sys   lhs,  sys rhs  )
          | Check     ( pred  pred, sys cont, srcinfo srcinfo )
          | NullSys   ()

  pred    = GTZ   ( affine expr )
          | GEZ   ( affine expr )
          | EQZ   ( affine expr )
          | Rel   ( sym name, sym* args )
          | Conj  ( pred lhs, pred rhs )
          | Disj  ( pred lhs, pred rhs )

  affine  = ( fraction offset, term* terms )
  term    = ( fraction coeff,  sym   var   )
}
""", {
  'sym':      lambda x: type(x) is Sym,
  'fraction': lambda x: type(x) is Fraction,
  'srcinfo':  lambda x: type(x) is SrcInfo,
})
ADTmemo(BD,['NullSys'])
BD.null = BD.NullSys()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Operator Overloading to help construct affine expressions

def _alift_(obj):
  if type(obj) is BD.affine:
    return obj
  elif type(obj) is BD.term:
    return BD.affine( Fraction(0), [obj] )
  elif type(obj) is Sym:
    return BD.affine( Fraction(0), [ BD.term(Fraction(1),obj) ] )
  elif type(obj) is int:
    return BD.affine( Fraction(obj), [] )
  elif type(obj) is Fraction:
    return BD.affine( obj, [] )
  else: assert False, f"unsupported affine lifting for type {type(obj)}"

@extclass(BD.affine)
def __add__(lhs,rhs):
  rhs   = _alift_(rhs)
  off   = lhs.offset + rhs.offset
  terms = lhs.terms.copy()
  terms.extend(rhs.terms)
  return BD.affine( off, terms )
@extclass(BD.affine)
def __radd__(rhs,lhs):  return _alift_(lhs) + rhs
@extclass(BD.affine)
def __neg__(arg):
  off   = -arg.offset
  terms = [ BD.term( -t.coeff, t.var ) for t in arg.terms ]
  return BD.affine( off, terms )
@extclass(BD.affine)
def __sub__(lhs,rhs):   return lhs + (-_alift_(rhs))
@extclass(BD.affine)
def __rsub__(rhs,lhs):  return _alift_(lhs) + (-rhs)

@extclass(BD.affine)
def __mul__(lhs,rhs):
  assert type(rhs) is Fraction, "expected fraction to scale by"
  off   = lhs.offset * rhs
  terms = [ BD.term( t.coeff * rhs, t.var ) for t in lhs.terms ]
  return BD.affine( off, terms )
@extclass(BD.affine)
def __rmul__(rhs,lhs): return rhs * lhs

@extclass(BD.affine)
def __gt__(lhs,rhs):
  return BD.GTZ( lhs - rhs )
@extclass(BD.affine)
def __ge__(lhs,rhs):
  return BD.GEZ( lhs - rhs )
@extclass(BD.affine)
def __lt__(lhs,rhs):
  return BD.GTZ( rhs - lhs )
@extclass(BD.affine)
def __le__(lhs,rhs):
  return BD.GEZ( rhs - lhs )
@extclass(BD.affine)
def eq(lhs,rhs):
  return BD.EQZ( lhs - rhs )

del __add__, __radd__, __neg__, __sub__, __rsub__, __mul__, __rmul__
del __gt__, __lt__, __ge__, __le__, eq

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Extraction of Bounds-Checking Problem

class BoundsExtraction:
  def __init__(self, f):
    self._ctxt  = Context()

    # pack context with relation sizes
    for rd in f.relations:
      self._ctxt.set(rd.name, rd.sizes)

    # construct the system corresponding to the expression body
    sys         = self.extract(f.body)

    # and then wrap that in variable declarations for sizes/relations
    for rd in reversed(f.relations):
      sys       = BD.RelIntro( rd.name, len(rd.sizes), sys, rd.srcinfo )
    for sz in reversed(f.sizes):
      sys       = BD.VarIntro( sz.name, sys, sz.srcinfo )

    self._sys   = sys


  def system(self): return self._sys

  def _get_rel_sizes(self,rname):
    szs = self._ctxt.get(rname)
    assert szs is not None, "Expected all relation lookups to succeed"
    return szs

  def extract(self, e):
    eclass  = type(e)

    # do not perform bounds-checks on code that was already checked.
    if hasattr(e, 'func_call_sub'):
      return BD.null

    elif eclass is AST.Var or eclass is AST.Const:
      return BD.null

    elif eclass is AST.BinOp:
      lhs = self.extract(e.lhs)
      rhs = self.extract(e.rhs)
      if   lhs is BD.null: return rhs
      elif rhs is BD.null: return lhs
      else:                return BD.Both(lhs,rhs)

    elif eclass is AST.Tuple:
      sys   = BD.null
      for a in e.args:
        s   = self.extract(a)
        if sys is BD.null:
          sys = s
        elif s is not BD.null:
          sys = BD.Both( s, sys )
      return sys

    elif eclass is AST.Proj:
      return self.extract(e.arg)

    elif eclass is AST.TensorLit:
      sys   = BD.null
      for a in e.args:
        s   = self.extract(a)
        if sys is BD.null:
          sys = s
        elif s is not BD.null:
          sys = BD.Both( s, sys )
      return sys

    elif eclass is AST.Gen or eclass is AST.Sum:
      # a sanity check; maybe not necessary, but if true
      # this greatly simplifies naming issues, so let's assert for now
      assert self._ctxt.get(e.name) is None, "Expected unique name symbols"
      # build guard predicate (let x be e.name)
      lo_bd = (_alift_(e.name) >= 0)
      hi_bd = (_alift_(e.name) < e.range)
      guard = BD.Conj( lo_bd, hi_bd )
      # assemble system
      body  = self.extract(e.body)
      if body is BD.null:
        return body
      else:
        body  = BD.Guard( guard, body, e.srcinfo )
        return  BD.VarIntro( e.name, body, e.srcinfo )

    elif eclass is AST.Access:
      # handle multiple accesses and w.r.t the tensor type...
      sys   = self.extract(e.base)
      typ   = e.base.type
      for i_ast in e.idx:
        i       = self.index(i_ast)
        rng     = typ.range
        typ     = typ.type
        lo_chk  = (i >= 0)
        hi_chk  = (i < rng)
        sys     = BD.Check( BD.Conj(lo_chk, hi_chk), sys, e.srcinfo )
      return sys

    elif eclass is AST.BuiltIn:
      # the built-in itself has no effect
      sys   = BD.null
      for a in e.args:
        s   = self.extract(a)
        if sys is BD.null:
          sys = s
        elif s is not BD.null:
          sys = BD.Both( s, sys )
      return sys

    elif eclass is AST.Indicate:
      # relational predicates may require introducing
      # additional bounds checks on their arguments
      guard, checks = self.pred(e.pred)
      body          = self.extract(e.body)
      if body is not BD.null:
        body        = BD.Guard( guard, body, e.srcinfo )
      # now wrap with any checks regardless of whether body is null
      for name,eq,chk,srcinfo in checks:
        body = BD.VarIntro( name,
                            BD.Guard( eq,
                                      BD.Check(chk,body,srcinfo),
                                      srcinfo ),
                            srcinfo )
      return body

    elif eclass is AST.Let:
      sys   = self.extract(e.ret)
      for s in e.stmts:
        rhs = self.extract(s.rhs)
        if sys is BD.null:
          sys = rhs
        elif rhs is not BD.null:
          sys = BD.Both( rhs, sys )
      return sys

    else: assert False, "unexpected case"

  def index(self, e):
    eclass  = type(e)
    if   eclass is AST.IdxConst:
      return _alift_(e.val)
    elif eclass is AST.IdxVar or eclass is AST.IdxSize:
      return _alift_(e.name)
    elif eclass is AST.IdxAdd:
      return self.index(e.lhs) + self.index(e.rhs)
    elif eclass is AST.IdxSub:
      return self.index(e.lhs) - self.index(e.rhs)
    elif eclass is AST.IdxScale:
      return e.coeff * self.index(e.idx)
    else: assert False, "unexpected case"

  # returns a second "checks" list of type
  #     [(var_name, var_eq, var_bd_chk, err_msg, srcinfo)]
  def pred(self, p):
    pclass  = type(p)
    if   pclass is AST.Cmp:
      lhs   = self.index(p.lhs)
      rhs   = self.index(p.rhs)
      if   p.op == "<":   return lhs < rhs, []
      elif p.op == ">":   return lhs > rhs, []
      elif p.op == "<=":  return lhs <= rhs, []
      elif p.op == ">=":  return lhs >= rhs, []
      elif p.op == "==":  return lhs.eq(rhs), []
      else: assert False, f"Unrecognized Op {p.op}"
    elif pclass is AST.Relation:
      sizes         = self._get_rel_sizes(p.name)
      args, checks  = [], []
      for k,(i_arg,N) in enumerate(zip(p.args,sizes)):
        i       = self.index(i_arg)
        v       = Sym(f"{p.name}{k}")
        def_eq  = i.eq(v)
        bd_chk  = BD.Conj( i >= 0, i < N )
        args.append(v)
        checks.append( (v,def_eq,bd_chk,i_arg.srcinfo) )
      return BD.Rel(p.name, args), checks

    elif pclass is AST.Conj or pclass is AST.Disj:
      lhs, lchk   = self.pred(p.lhs)
      rhs, rchk   = self.pred(p.rhs)
      lchk.extend(rchk)
      ctr         = BD.Conj if pclass is AST.Conj else BD.Disj
      return ctr(lhs,rhs), lchk

    else: assert False, "Impossible Case"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Execution of Bounds-Checking Problem

class BoundsCheck:
  def __init__(self,sys):
    assert isinstance(sys, BD.sys), "Expected a bounds-system..."
    self._slv   = _get_smt_solver()
    self._slv.reset_assertions()

    self._ctxt    = Context()
    self._errors  = []

    self._slv.push()
    self.check(sys)
    self._slv.pop()

    self.report_errors()

  def report_errors(self):
    if len(self._errors) == 0: return

    errs = []
    errs.append("Failed Bounds Checks:")
    for srcinfo,msg in self._errors:
      errs.append(f"{srcinfo}: {msg}")
    raise TypeError("\n".join(errs))

  def _err(self, node, msg):
    self._errors.append((node.srcinfo, msg))

  def _get_solution(self, pred):
    smt_syms  = [ smt_sym for nm,smt_sym in self._ctxt.items()
                          if smt_sym.get_type() == SMT.INT ]
    self._slv.push()
    self._slv.add_assertion(pred)
    val_map = self._slv.get_py_values(smt_syms)
    self._slv.pop()
    mapping   = []
    for nm,smt_sym in self._ctxt.items():
      if smt_sym.get_type() == SMT.INT:
        mapping.append(f"  {nm} = {val_map[smt_sym]}")
    return "\n".join(mapping)


  def check(self, sys):
    styp = type(sys)
    if   styp is BD.VarIntro:
      smtsym  = SMT.Symbol(repr(sys.name), SMT.INT)
      self._ctxt.set(sys.name, smtsym)
      self.check(sys.cont)

    elif styp is BD.RelIntro:
      Rtyp    = SMT.FunctionType(SMT.BOOL,
                                 [ SMT.INT for i in range(0,sys.n_args) ])
      smtsym  = SMT.Symbol(repr(sys.name), Rtyp)
      self._ctxt.set(sys.name, smtsym)
      self.check(sys.cont)

    elif styp is BD.Guard:
      pred    = self.formula(sys.pred)
      self._slv.add_assertion(pred)
      self.check(sys.cont)

    elif styp is BD.Both:
      # make sure we can backtrack from the first branch
      self._slv.push()
      self._ctxt.push()
      self.check(sys.lhs)
      self._ctxt.pop()
      self._slv.pop()
      # now the second branch we can just proceed
      self.check(sys.rhs)

    elif styp is BD.Check:
      pred    = SMT.Not( self.formula(sys.pred) )
      failure = self._slv.is_sat(pred)
      if failure:
        mapping = self._get_solution(pred)
        self._err(sys, f"Out of Bounds Access:\n{mapping}")
      # continue regardless
      self.check(sys.cont)

    elif styp is BD.NullSys:
      pass

  def formula(self, p):
    ptyp = type(p)
    if ptyp is BD.GTZ or ptyp is BD.GEZ or ptyp is BD.EQZ:
      aform   = self.affine(p.expr)
      if   ptyp is BD.GTZ:  return SMT.GT( aform, SMT.Int(0) )
      elif ptyp is BD.GEZ:  return SMT.GE( aform, SMT.Int(0) )
      elif ptyp is BD.EQZ:  return SMT.Equals( aform, SMT.Int(0) )
      else: assert False

    elif ptyp is BD.Rel:
      rsym    = self._ctxt.get(p.name)
      assert rsym is not None, f"expected relation name '{p.name}'"

      args    = []
      for a in p.args:
        sym   = self._ctxt.get(a)
        assert sym is not None, f"expected variable name '{a}'"
        args.append(sym)
      return  SMT.Function(rsym,args)

    elif ptyp is BD.Conj or ptyp is BD.Disj:
      lhs     = self.formula(p.lhs)
      rhs     = self.formula(p.rhs)
      smtop   = SMT.And if ptyp is BD.Conj else SMT.Or
      return smtop(lhs, rhs)

  def affine(self, a):
    """ return some positive rescaling of the affine expression
        s.t. the rescaled expression has integer coefficients
        safe, since positive rescaling preserves
        all of a >= 0, a > 0, and a == 0 """
    # find the lcm of the offset denominator
    # and all coefficient denominators
    mult    = a.offset.denominator
    for t in a.terms:
      mult  = _lcm(mult, t.coeff.denominator)
    # now, we can produce an integral affine equation,
    # by rescaling through with `mult`
    a = a * Fraction(mult)
    # Finally, convert this to an SMT formula
    assert a.offset.denominator == 1
    f       = SMT.Int(a.offset.numerator)
    for t in a.terms:
      assert t.coeff.denominator == 1
      sym   = self._ctxt.get(t.var)
      assert sym is not None, f"expected variable name '{t.var}'"
      term  = SMT.Times( SMT.Int(t.coeff.numerator), sym )
      f     = SMT.Plus( f, term )
    return f



