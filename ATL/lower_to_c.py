
from .prelude import *
from . import builtins as B
from . import atl_types as T

from .frontend import AST

from collections import ChainMap, namedtuple
import itertools
import math
from fractions import Fraction
import re

import ctypes
import os
import sys
import time
import subprocess

import hashlib

import numpy as np

from .c_lang import C, compile_procs
from .c_lang import get_type as C_get_type

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Helpers and directory setup

def _shell(cstr):
  subprocess.run(cstr, check=True, shell=True)

_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_C_CACHE        = os.path.join(_HERE_DIR,'.atl_c_cache')
if not os.path.isdir(_C_CACHE):
  os.mkdir(_C_CACHE)

def get_time(s):
  if not os.path.exists(s):
    return None
  else:
    return os.path.getmtime(s)

# clean the cache when it exceeds 50MB;
# keep all files created in the last day or so
def clean_cache(size_trigger = int(50e6), clear_time_window = 86400.0):
  curr_time   = time.time()
  filenames   = [ os.path.join(_C_CACHE,f) for f in os.listdir(_C_CACHE) ]
  used_size   = sum(os.path.getsize(f) for f in filenames)

  if used_size > size_trigger:
    for file in filenames:
      mtime   = get_time(file)
      if curr_time - mtime > clear_time_window:
        os.remove(file)

# try out a single clean every time we load this module
clean_cache()


def subst_rhs(e, env):
  etyp = type(e)
  if etyp is AST.Var:
    return env[e.name] if e.name in env else e
  elif etyp is AST.Const:
    return e
  elif etyp is AST.BinOp:
    return AST.BinOp(e.op, subst_rhs(e.lhs, env), subst_rhs(e.rhs, env),
                     e.type, e.srcinfo)
  elif etyp is AST.Tuple or etyp is AST.TensorLit:
    return etyp([ subst_rhs(a, env) for a in e.args ], e.type, e.srcinfo)
  elif etyp is AST.Proj:
    return AST.Proj(e.idx, subst_rhs(e.arg, env), e.type, e.srcinfo)
  elif etyp is AST.Gen or etyp is AST.Sum:
    return etyp(e.name, e.range, subst_rhs(e.body, env), e.type, e.srcinfo)
  elif etyp is AST.Access:
    return AST.Access(subst_rhs(e.base, env), e.idx, e.type, e.srcinfo)
  elif etyp is AST.BuiltIn:
    return etyp(e.f, [ subst_rhs(a, env) for a in e.args ], e.type, e.srcinfo)
  elif etyp is AST.Indicate:
    return AST.Indicate(e.pred, subst_rhs(e.body, env), e.type, e.srcinfo)
  elif etyp is AST.Let:
    # short-cut for where I want to use this
    return e
  else: assert False, "bad case"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class LVal:
  def __init__(self, base_var, idx=[], proj=[], op="=", is_out=False):
    self._name  = base_var
    self._idx   = idx
    self._proj  = proj
    self._op    = op
    self._is_out= is_out

  def idx(self, e):
    return LVal(self._name, self._idx + [e], self._proj, self._op,
                self._is_out)

  def proj(self, i):
    return LVal(self._name, self._idx, self._proj + [i], self._op,
                self._is_out)

  def reduce(self, is_out=None):
    is_out = self._is_out if is_out is None else is_out
    return LVal(self._name, self._idx, self._proj, "+=", is_out)

  def acc(self):
    return C.access( self._name, self._idx, self._proj )

  def op(self):
    return self._op

  def is_out(self):
    return self._is_out



class Lower:
  """ Idea of pass is to implement a core function that translates
        (lval, expr) ---> statment
      where lval is some kind of access expression, plus
      whether the expression on the right should be reduced into
      the lval or simply assigned to the lvalue.
  """

  def __init__(self, func):
    self._func = func
    #print(func)

    sizes     = [ sz.name for sz in func.sizes ]
    args      = [ C.fnarg(a.name, a.type, a.srcinfo) for a in func.vars ]
    relargs   = [ C.relarg(r.name, tuple(r.sizes), r.srcinfo)
                  for r in func.relations ]

    assert all( str(a.name) != 'output' for a in args )
    output  = Sym('output')
    args   += [ C.fnarg(output, func.body.type, func.srcinfo) ]
    body    = self.special_case_output(LVal(output, is_out=True), func.body)

    self._proc = C.proc( func.name or "anon",
                         sizes,
                         args,
                         relargs,
                         body,
                         func.srcinfo )

  def result(self):
    return self._proc

  def special_case_output(self, lval, body):
    # check if we have a special case where the output is simply
    # packing data into a struct.
    def check_output(e):
      if type(e) is AST.Tuple:
        return all(check_output(a) for a in e.args)
      else:
        # assert that we never pack an input directly in this case
        assert type(e) is not AST.Var or e.name not in self._func.arg_order
        return type(e) is AST.Var
    
    if type(body) is AST.Let and check_output(body.ret):
      # then substitute the relevant output name in for the
      # intermediate variable name everywhere
      lvals   = {}
      rvars   = {}
      def build_out(lval, rvar, e):
        if type(e) is AST.Tuple:
          for i,a in enumerate(e.args):
            build_out(lval.proj(i),
                      AST.Proj(i, rvar, rvar.type.types[i], rvar.srcinfo),
                      a)
        else:
          assert type(e) is AST.Var
          lvals[e.name]   = lval
          rvars[e.name]   = rvar
      build_out(lval,
                AST.Var(lval._name, body.type, body.srcinfo),
                body.ret)

      # custom let processing
      stmts   = []
      for s in body.stmts:
        rhs = subst_rhs(s.rhs, rvars)
        if s.name in lvals: # bind directly to output if appropriate
          stmts += self.lower(lvals[s.name], rhs)
        else: # standard case w/ rhs substitution
          stmts.append( C.Alloc(s.name, s.type, s.srcinfo) )
          stmts += self.lower(LVal(s.name), rhs)
      # no need to bind return output further
      return stmts

    # fall-back is to just do normal lowering to the whole body
    else:
      return self.lower(lval, body)

  def lower(self, lval, e, is_out=False):
    """ returns a list of statements """
    etyp  = type(e)

    # leaves
    if etyp is AST.Var or etyp is AST.Proj or etyp is AST.Access:
      return [ self.assemble(lval, self.lower_leaf(e)) ]
    elif etyp is AST.Const:
      rhs = C.Const(e.val, float, e.srcinfo)
      return [ self.assemble(lval, rhs) ]

    # computation
    elif etyp is AST.BinOp:
      assert e.type == T.num
      left  = LVal(Sym('t'))
      right = LVal(Sym('t'))
      rhs   = C.BinOp(e.op, C.Read(left.acc(),  float, e.srcinfo),
                            C.Read(right.acc(), float, e.srcinfo),
                            float, e.srcinfo)
      return ([ C.Alloc(left._name, T.num, e.srcinfo),
                C.Alloc(right._name, T.num, e.srcinfo) ] + 
              self.lower(left, e.lhs) +
              self.lower(right, e.rhs) +
              [ self.assemble(lval, rhs) ])
    elif etyp is AST.BuiltIn:
      xs    = [ Sym('t') for _ in e.args ]
      stmts = [ C.Alloc(x, T.num, e.srcinfo) for x in xs ]
      for x,a in zip(xs, e.args):
        stmts += self.lower(LVal(x), a)
      args  = [ C.Read(LVal(x).acc(), float, e.srcinfo) for x in xs ]
      rhs   = C.BuiltIn(e.f, args, float, e.srcinfo)
      return stmts + [ self.assemble(lval, rhs) ]

    # constructors
    elif etyp is AST.Tuple:
      stmts = []
      for i,a in enumerate(e.args):
        stmts += self.lower(lval.proj(i), a)
      return stmts
    elif etyp is AST.TensorLit:
      stmts = []
      for i,a in enumerate(e.args):
        idx   = C.Const(i, int, e.srcinfo)
        stmts += self.lower(lval.idx(idx), a)
      return stmts
    elif etyp is AST.Gen:
      idx     = C.Read( C.access(e.name,[],[]), int, e.srcinfo )
      lo, hi  = self.build_loop_limits(e.range, e)
      body    = self.lower(lval.idx(idx), e.body)
      return [ C.For(e.name, lo, hi, body, e.srcinfo) ]

    # Other
    elif etyp is AST.Sum:
      assert e.type == T.num
      if lval.is_out():
        zero_out = [ C.Assign( lval.acc(),
                               C.Const(0.0, float, e.srcinfo),
                               e.srcinfo) ]
      else:
        zero_out = []
      lo, hi  = self.build_loop_limits(e.range, e)
      body    = self.lower(lval.reduce(is_out=False), e.body)
      return zero_out + [ C.For(e.name, lo, hi, body, e.srcinfo) ]
    elif etyp is AST.Indicate:
      cond    = self.lower_pred(e.pred)
      body    = self.lower(lval, e.body)
      return [ C.If(cond, body, e.srcinfo) ]

    # Binding
    elif etyp is AST.Let:
      stmts   = []
      for s in e.stmts:
        stmts.append( C.Alloc(s.name, s.type, s.srcinfo) )
        stmts += self.lower(LVal(s.name), s.rhs)
      stmts  += self.lower(lval, e.ret)
      return stmts

    else: assert False, "bad case"

  def assemble(self, lval, rhs):
    Op    = C.Assign if lval.op() == '=' else C.Reduce
    return Op(lval.acc(), rhs, rhs.srcinfo)

  def lower_leaf(self, e):
    orig_e = e
    srcinfo = e.srcinfo
    assert e.type == T.num
    # unpack any accesses
    idx   = []
    while type(e) is AST.Access:
      idx = [ self.lower_aff(i) for i in e.idx ] + idx
      e   = e.base

    # unpack any projections
    proj  = []
    while type(e) is AST.Proj:
      proj.append(e.idx)
      e   = e.arg
    proj  = list(reversed(proj))

    # now we should have a variable leftover
    assert type(e) is AST.Var
    return C.Read(C.access(e.name, idx, proj), float, srcinfo)

  def build_loop_limits(self, rng, e):
    zero  = C.Const( 0, int, e.srcinfo )
    if type(rng) is int:
      hi  = C.Const( rng, int, e.srcinfo )
    else:
      hi  = C.Read( C.access(rng,[],[]), int, e.srcinfo )
    return zero, hi

  def lower_aff(self, a):
    atyp  = type(a)
    if atyp is AST.IdxConst:
      return C.Const(a.val, int, a.srcinfo)
    elif atyp is AST.IdxVar or atyp is AST.IdxSize:
      return C.Read( C.access(a.name,[],[]), int, a.srcinfo )
    elif atyp is AST.IdxAdd or atyp is AST.IdxSub:
      op = "+" if atyp is AST.IdxAdd else "-"
      return C.BinOp(op, self.lower_aff(a.lhs),
                         self.lower_aff(a.rhs), int, a.srcinfo)
    elif atyp is AST.IdxScale:
      assert int(a.coeff) == a.coeff, "TODO? handle actual fractions"
      return C.BinOp('*', C.Const(int(a.coeff), int, a.srcinfo),
                          self.lower_aff(a.idx),
                          int, a.srcinfo)
    else: assert False, "bad case"

  def lower_pred(self, p):
    ptyp  = type(p)
    if ptyp is AST.Cmp:
      return C.BinOp(p.op, self.lower_aff(p.lhs),
                           self.lower_aff(p.rhs), bool, p.srcinfo)
    elif ptyp is AST.Relation:
      idx = [ self.lower_aff(a) for a in p.args ]
      return C.Read( C.access(p.name,idx,[]), bool, p.srcinfo )
    elif ptyp is AST.Conj or ptyp is AST.Disj:
      op = 'and' if ptyp is AST.Conj else 'or'
      return C.BinOp(op, self.lower_pred(p.lhs),
                         self.lower_pred(p.rhs), bool, p.srcinfo)
    else: assert False, "bad case"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

from .cjit import CJitSig
from .cjit import CJit as GenericCJit

class CJit:
  """ Manage JIT compilation of ATL -> C code
  """

  def __init__(self, func):
    assert type(func) is AST.function

    self._func    = func
    self._proc    = Lower(func).result()


    name          = self._proc.name
    cstr          = compile_procs(None, None, [self._proc])
    self._c_code_str = cstr
    signature = CJitSig(
      size_names  = [x.name for x in self._func.sizes],
      var_typs    = [x.type for x in self._func.vars],
      n_rels      = len(self._func.relations),
      ret_typ     = self._func.rettype)
    self._generic_cjit  = GenericCJit(name, signature,
                                      cstr,
                                      f"-I {_C_CACHE}")

  def codestr(self):
    return self._c_code_str

  def __call__(self, vs, sizes, relations, out):

    self._generic_cjit(vs, sizes, relations, out)



