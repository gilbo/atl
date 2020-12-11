
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
  def __init__(self, base_var, idx=[], proj=[], op="="):
    self._name  = base_var
    self._idx   = idx
    self._proj  = proj
    self._op    = op

  def idx(self, e):
    return LVal(self._name, self._idx + [e], self._proj, self._op)

  def proj(self, i):
    return LVal(self._name, self._idx, self._proj + [i], self._op)

  def reduce(self):
    return LVal(self._name, self._idx, self._proj, "+=")

  def acc(self):
    return C.access( self._name, self._idx, self._proj )

  def op(self):
    return self._op


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
    body    = self.special_case_output(LVal(output), func.body)

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


  def lower(self, lval, e):
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
      lo, hi  = self.build_loop_limits(e.range, e)
      body    = self.lower(lval.reduce(), e.body)
      return [ C.For(e.name, lo, hi, body, e.srcinfo) ]
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

_ctype_struct_cache = {}
def get_ctype(typ):
  """ translate double,bool,int or an ATL type into a C-type """
  if typ == float or typ == T.num:
    return ctypes.c_double
  elif typ == int:
    return ctypes.c_int
  elif typ == bool:
    return ctypes.c_bool
  elif type(typ) is T.Tensor:
    assert typ.is_SoA()
    return ctypes.POINTER(ctypes.c_double)
  elif type(typ) is T.Tuple:
    if typ in _ctype_struct_cache:
      return _ctype_struct_cache[typ]

    name, defstr  = C_get_type(typ)
    fields        = [ (f"_{i}", get_ctype(t))
                      for i,t in enumerate(typ.types) ]
    # create a new type
    struct        = type(name, (ctypes.Structure,), { '_fields_': fields })

    _ctype_struct_cache[typ] = struct
    return struct
  else: assert False, "bad case"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class CJit:
  """ Manage JIT compilation of ATL -> C code
  """

  def __init__(self, func):
    assert type(func) is AST.function

    self._func    = func
    self._proc    = Lower(func).result()

    # run once just to compute hash values...
    hstr, cstr    = compile_procs("dummy.h", "dummy.c", [self._proc])
    hashstr       = hashlib.md5(cstr.encode('utf-8')).hexdigest()

    name          = self._proc.name
    fname         = name + hashstr
    h_filename    = os.path.join(_C_CACHE,f"{fname}.h")
    c_filename    = os.path.join(_C_CACHE,f"{fname}.cpp")
    so_filename   = os.path.join(_C_CACHE,f"{fname}.so")
    comp_cmd      = (f"clang++ -Wall -Werror -fPIC -O3 -shared -std=c++11 "
                     f"-I {_C_CACHE} "
                     f"-o {so_filename} {c_filename}")
    hstr, cstr    = compile_procs(h_filename, c_filename, [self._proc])

    #print(hstr)
    #print(cstr)

    def matches_file(src, fname):
      if not os.path.isfile(fname):
        return False
      else:
        with open(fname, 'r', encoding = 'utf-8') as F:
          return F.read() == src
    def write_file(src, fname):
      with open(fname, 'w', encoding = 'utf-8') as F:
        F.write(src)

    # do we need to rebuild the corresponding SO?
    if (not matches_file(hstr, h_filename) or 
        not matches_file(cstr, c_filename)):
      write_file(hstr, h_filename)
      write_file(cstr, c_filename)
      #print(comp_cmd)
      _shell(comp_cmd)

    # load the module regardless
    module        = ctypes.CDLL(so_filename)
    self._module  = module
    self._cfun    = getattr(module, name)

    #wrap the module function with types
    atyps = []
    for sz in self._proc.sizes:
      atyps.append(get_ctype(int))
    for a in self._proc.args:
      if str(a.name) == "output":
        atyps.append(ctypes.POINTER(get_ctype(a.type)))
      else:
        atyps.append(get_ctype(a.type))
    for sz in self._proc.relargs:
      atyps.append(ctypes.POINTER(ctypes.c_bool))
    self._cfun.argtypes   = atyps
    self._cfun.restype    = None

  def __call__(self, vs, sizes, relations, out):

    def pack_input(val,typ,is_output=False):
      if typ == T.num:
        if is_output:
          assert type(val) is np.ndarray and val.shape == (1,)
          return ctypes.c_double(val[0])
        else:
          assert type(val) is float
          return ctypes.c_double(val)
      elif type(typ) is T.Tensor:
        assert type(val) is np.ndarray
        ptr       = val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return ptr
      elif type(typ) is T.Tuple:
        assert isinstance(val, tuple) and len(val) == len(typ.types)
        struct    = get_ctype(typ)
        return struct(*[ pack_input(v,t,is_output)
                         for v,t in zip(val,typ.types) ])
      else: assert False, "bad case"

    # collect all the arguments to the function call
    fargs         = []
    for sz in sizes:
      fargs.append(ctypes.c_int(sz))
    for v,vd in zip(vs, self._func.vars):
      fargs.append(pack_input(v,vd.type))
    # special case the packing of the output argument...
    out_obj = pack_input(out, self._func.rettype, is_output=True)
    fargs.append(ctypes.byref(out_obj))
    # back to the relation data now...
    for r in relations:
      assert type(r) is np.ndarray
      fargs.append( r.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)) )

    # do the C-function invocation through the FFI
    self._cfun(*fargs)

    def pack_output(obj,argval,typ):
      if typ == T.num:
        assert type(argval) is np.ndarray and argval.shape == (1,)
        #print(obj)
        #print(type(obj))
        if type(obj) is float:
          argval[0] = obj
        else:
          argval[0] = obj.value
      elif type(typ) is T.Tensor:
        assert type(argval) is np.ndarray
        pass # no action necessary, data written via pointer already
      elif type(typ) is T.Tuple:
        for i,(a,t) in enumerate(zip(argval,typ.types)):
          pack_output( getattr(obj,f"_{i}"), a, t )
      else: assert False, "bad case"

    pack_output(out_obj, out, self._func.rettype)








