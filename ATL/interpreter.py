
#from adt import ADT
#from adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from .frontend import AST

from fractions import Fraction

import numpy as np

import math

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def Interpret(ast, vs, sizes, relations, output):
  _Interpreter(ast, vs, sizes, relations, output)

class _Interpreter:
  def __init__(self, ast, vs, sizes, relations, output):
    assert len(ast.vars) == len(vs)
    assert len(ast.sizes) == len(sizes)
    assert len(ast.relations) == len(relations)

    self._ctxt    = Context()

    # bind all inputs
    for sd,val in zip(ast.sizes, sizes):
      self._ctxt.set(sd.name, val)
    for vd,val in zip(ast.vars, vs):
      self._ctxt.set(vd.name, self._convert_input(vd.type, val))
    for rd,val in zip(ast.relations, relations):
      self._ctxt.set(rd.name, self._convert_rel_input(rd.sizes, val))

    out           = self._exec(ast.body)
    self._write_output(ast.body.type, out, output)

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

  def _write_output(self, typ, val, buf):
    assert typ.is_SoA()
    if typ is T.num:
      buf[0] = val
    elif type(typ) is T.Tensor:
      N     = self._get_range(typ.range)
      if typ.type is T.num:
        for i in range(0,N):
          buf[i] = val[i]
      else:
        for i in range(0,N):
          self._write_output(typ.type, val[i], buf[i])
    elif type(typ) is T.Tuple:
      assert len(typ.types) == len(val)
      assert len(val) == len(buf)
      for t,v,b in zip(typ.types,val,buf):
        self._write_output(t,v,b)

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

  def _make_zero(self, typ):
    if typ is T.num:
      return 0
    elif type(typ) is T.Tuple:
      return tuple( _make_zero(t) for t in typ.types )
    elif type(typ) is T.Tensor:
      N     = self._get_range(typ.shape()[0])
      return [ _make_zero(typ.type) for _ in range(0,N) ]
    else: assert False, "impossible case"

  def _get_val(self,nm):
    return self._ctxt.get(nm)

  def _get_range(self,rng):
    if type(rng) is int:  return rng
    else:                 return self._ctxt.get(rng)

  def _exec(self,e):
    eclass    = type(e)

    #   Expr Exec   #

    if   eclass is AST.Var:
      return self._get_val(e.name)
    elif eclass is AST.Const:
      return e.val
    elif eclass is AST.Add:
      return self._Add_shapes( e.lhs.type, self._exec(e.lhs),
                                           self._exec(e.rhs) )
    elif eclass is AST.Mul:
      return self._exec(e.lhs) * self._exec(e.rhs)
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
      for i in range(0,N):
        self._ctxt.set(e.name, i)
        result = self._Add_shapes(e.type, result, self._exec(e.body))
      self._ctxt.pop()
      return result
    elif eclass is AST.Access:
      base    = self._exec(e.base)
      for ie in e.idx:
        i = math.floor( self._exec(ie) )
        base  = base[i]
      return base
    elif eclass is AST.BuiltIn:
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
      return Fraction( self._get_val(e.name) )
    elif eclass is AST.IdxAdd:
      return self._exec(e.lhs) + self._exec(e.rhs)
    elif eclass is AST.IdxScale:
      return e.coeff * self._exec(e.idx)

    #   Pred Exec   #

    elif eclass is AST.Cmp:
      lhs, rhs  = self._exec(e.lhs), self._exec(e.rhs)
      if   e.op == '==':  return (lhs == rhs)
      elif e.op == '<':   return (lhs <  rhs)
      elif e.op == '>':   return (lhs >  rhs)
      elif e.op == '<=':  return (lhs <= rhs)
      elif e.op == '>=':  return (lhs >= rhs)
      else: assert False, "invalid op"
    elif eclass is AST.Relation:
      R       = self._get_val(e.name)
      for ie in e.args:
        i = math.floor( self._exec(ie) )
        R     = R[i]
      return R
    elif eclass is AST.Conj or eclass is AST.Disj:
      lhs, rhs  = self._exec(e.lhs), self._exec(e.rhs)
      if   eclass is AST.Conj: return lhs and rhs
      elif eclass is AST.Disj: return lhs or rhs
      else: assert False, "impossible case"


