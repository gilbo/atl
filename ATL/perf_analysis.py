
from .prelude import *

from . import atl_types as T
from .frontend import AST

from fractions import Fraction

#import numpy as np

#import math

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Analysis:
  def __init__(self, ast, vs, sizes, relations):
    assert len(ast.vars) == len(vs)
    assert len(ast.sizes) == len(sizes)
    assert len(ast.relations) == len(relations)

    self._ctxt    = Context()
    self._ast     = ast

    # bind all inputs
    for sd,val in zip(ast.sizes, sizes):
      self._ctxt.set(sd.name, val)
    for vd,val in zip(ast.vars, vs):
      self._ctxt.set(vd.name, self._convert_input(vd.type, val))
    for rd,val in zip(ast.relations, relations):
      self._ctxt.set(rd.name, self._convert_rel_input(rd.sizes, val))

    # profiling / analysis information
    self._op_counts = {
      "+"   :   0,
      "*"   :   0,
      "/"   :   0,
      "idx+":   0,
      "idx*":   0,
      "bool":   0,
    }

    out           = self._exec(ast.body)
    self._print_report()

  def _print_report(self):
    print(f"Perf Counts for {self._ast.name}")
    for nm,c in sorted(self._op_counts.items()):
      print(f"{nm:11}: {c:11}")

  def _convert_rel_input(self, sizes, val):
    if len(sizes) == 0:
      return val
    else:
      N     = self._get_range(sizes[0])
      assert val.shape[0] == N
      return [ self._convert_rel_input(sizes[1:], val[i])
               for i in range(0,N) ]

  def _convert_input(self, typ, val):
    if typ is T.num:
      return val
    elif type(typ) is T.Tuple:
      return tuple( self._convert_input(t,v)
                    for t,v in zip(typ.types,val) )
    elif type(typ) is T.Tensor:
      assert typ.is_SoA()
      N     = self._get_range(typ.shape()[0])
      assert val.shape[0] == N
      return [ self._convert_input(typ.type, val[i])
               for i in range(0,N) ]
    else: assert False, "impossible case"

  def _Add_shapes(self, typ, lhs, rhs):
    if typ is T.num:
      return lhs + rhs
    elif type(typ) is T.Tuple:
      return tuple( self._Add_shapes(t,l,r)
                    for t,l,r in zip(typ.types,lhs,rhs) )
    elif type(typ) is T.Tensor:
      return [ self._Add_shapes(typ.type,l,r)
               for l,r in zip(lhs,rhs) ]
    else: assert False, "impossible case"

  def _count_shape(self, typ):
    if typ is T.num:
      return 1
    elif type(typ) is T.Tuple:
      return sum([ self._count_shape(t) for t in typ.types ])
    elif type(typ) is T.Tensor:
      N     = self._get_range(typ.range)
      return self._count_shape(typ.type) * N

  def _make_zero(self, typ):
    if typ is T.num:
      return 0
    elif type(typ) is T.Tuple:
      return tuple( self._make_zero(t) for t in typ.types )
    elif type(typ) is T.Tensor:
      N     = self._get_range(typ.range)
      return [ self._make_zero(typ.type) for _ in range(0,N) ]
    else: assert False, "impossible case"

  def _get_val(self,nm):
    return self._ctxt.get(nm)

  def _get_range(self,rng):
    if type(rng) is int:  return rng
    else:                 return self._ctxt.get(rng)

  def _try_idx(self, base, idx):
    for i in idx:
      if i.denominator != 1:
        return None
      i   = i.numerator
      if i < 0 or i >= len(base):
        return None
      base = base[i]
    return base

  def _exec(self,e):
    eclass    = type(e)

    #   Expr Exec   #

    if   eclass is AST.Var:
      return self._get_val(e.name)

    elif eclass is AST.Const:
      return e.val

    elif eclass is AST.BinOp:
      lval    = self._exec(e.lhs)
      rval    = self._exec(e.rhs)
      n_shape = self._count_shape(e.lhs.type)
      def is_neg1(c): return type(c) is AST.Const and c.val == -1.0
      if e.op == '+' or e.op == '-':
        self._op_counts['+']  += n_shape
      elif e.op == '*' and (is_neg1(e.lhs) or is_neg1(e.rhs)):
        pass # don't count multiplication by constant negative 1
      else:
        self._op_counts[e.op] += n_shape

      if e.op == '+':
        return self._Add_shapes( e.lhs.type, lval, rval )
      elif e.op == '-': return lval - rval
      elif e.op == '*': return lval * rval
      elif e.op == '/': return lval / rval

    elif eclass is AST.Tuple:
      return tuple( self._exec(a) for a in e.args )

    elif eclass is AST.Proj:
      return ( self._exec(e.arg) )[e.idx]

    elif eclass is AST.TensorLit:
      return [ self._exec(a) for a in e.args ]

    elif eclass is AST.Gen:
      N       = self._get_range(e.range)
      result  = []
      self._ctxt.push()
      for i in range(0,N):
        self._ctxt.set(e.name, i)
        result.append( self._exec(e.body) )
      self._ctxt.pop()
      return result

    elif eclass is AST.Sum:
      N       = self._get_range(e.range)
      result  = self._make_zero(e.type)
      self._ctxt.push()
      n_shape = self._count_shape(e.type)

      for i in range(0,N):
        self._ctxt.set(e.name, i)
        self._op_counts['+']  += n_shape
        result = self._Add_shapes(e.type, result, self._exec(e.body))
      self._ctxt.pop()
      return result

    elif eclass is AST.Access:
      base    = self._exec(e.base)
      idx     = [ self._exec(ie) for ie in e.idx ]
      val     = self._try_idx(base,idx)
      return self._make_zero(e.type) if val is None else val

    elif eclass is AST.BuiltIn:
      nm      = e.f.name()
      if nm not in self._op_counts: self._op_counts[nm] = 0
      self._op_counts[nm] += 1

      args    = [ self._exec(a) for a in e.args ]
      return e.f.interpret(*args)

    elif eclass is AST.Indicate:
      guard   = self._exec(e.pred)
      if not guard:
        return self._make_zero(e.type)
      else:
        return self._exec(e.body)
    elif eclass is AST.Let:
      self._ctxt.push()
      for stmt in e.stmts:
        self._ctxt.set(stmt.name, self._exec(stmt.rhs))
      result  = self._exec(e.ret)
      self._ctxt.pop()
      return result

    #   Index Exec   #

    elif eclass is AST.IdxConst:
      return Fraction(e.val)
    elif eclass is AST.IdxVar or eclass is AST.IdxSize:
      x = self._get_val(e.name)
      return Fraction( x )
    elif eclass is AST.IdxAdd:
      self._op_counts["idx+"] += 1
      return self._exec(e.lhs) + self._exec(e.rhs)
    elif eclass is AST.IdxSub:
      self._op_counts["idx+"] += 1
      return self._exec(e.lhs) - self._exec(e.rhs)
    elif eclass is AST.IdxScale:
      self._op_counts["idx*"] += 1
      return e.coeff * self._exec(e.idx)

    #   Pred Exec   #

    elif eclass is AST.Cmp:
      self._op_counts["bool"] += 1
      lhs, rhs  = self._exec(e.lhs), self._exec(e.rhs)
      if   e.op == '==':  return (lhs == rhs)
      elif e.op == '<':   return (lhs <  rhs)
      elif e.op == '>':   return (lhs >  rhs)
      elif e.op == '<=':  return (lhs <= rhs)
      elif e.op == '>=':  return (lhs >= rhs)
      else: assert False, "invalid op"
    elif eclass is AST.Relation:
      R       = self._get_val(e.name)
      args    = [ self._exec(ie) for ie in e.args ]
      val     = self._try_idx(R,args)
      return val if val else False
    elif eclass is AST.Conj or eclass is AST.Disj:
      self._op_counts["bool"] += 1
      lhs, rhs  = self._exec(e.lhs), self._exec(e.rhs)
      if   eclass is AST.Conj: return lhs and rhs
      elif eclass is AST.Disj: return lhs or rhs
      else: assert False, "impossible case"
