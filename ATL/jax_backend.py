
from .prelude import *

from . import atl_types as T
from .py_type_values import get_python_named_tuple
from .frontend import AST

from collections import ChainMap, namedtuple
import itertools
import math
from fractions import Fraction
import re

import jax
import jax.numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

ATerm = namedtuple('ATerm',['c','x'])
Aff   = namedtuple('Aff',['off','terms'])

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class Compile:
  def __init__(self, ast):
    self._ast         = ast
    self._ctxt        = {}
    self._names       = {}

    self._body        = []
    self._shape       = tuple()
    self._out_idx     = tuple()
    self._locmap      = ChainMap({},locals())
    # bind local variables
    self.bind_local('_named_tuples', {})
    self.bind_local('np',            np)
    self.bind_local('jax',           jax)

    # assemble a string for the function...
    name, sig, out    = self.build_func_sig(ast)
    self._out_sym     = out
    self.comp_body(ast.body)

    # compile the function string...
    exec((f"{sig}\n"
          f"{'\n'.join(self._body)}"),
         globals(),locmap)
    fn                = locmap[name]
    fn                = jax.jit(fn)
    self._jax_fn      = fn


  def build_func_sig(self, ast):
    name      = ast.name or 'anon_func'
    args      = [ self.new_name(s) for s in ast.arg_order ]

    # reserve the name 'output'
    assert 'output' not in self._names
    output    = Sym('output')
    args.append(self.new_name(output))

    code_str  = f"def {name}({','.join(args)}):"

    return name, code_str, output

  def bind_local(self,key,val):
    """ make sure these values don't get shadowed """
    assert key not in self._names
    self._names[key] = val
    self._locmap[key] = val

  def new_name(self,nm):
    assert nm not in self._ctxt
    nmstr = str(nm)
    if nmstr not in self._names:
      # the first occurrence of nmstr is undecorated
      retstr = nmstr
      self._names[nmstr] = retstr
    else:
      # if we've already allocated nmstr, then
      # we must allocate a fresh name via a count decoration
      s   = self._names[nmstr]
      m = re.match('^(.*)_([0-9]*)$', s)
      if not m:
        # s was undecorated
        retstr = s + "_1"
      else:
        # increment the counter
        retstr = f"{m[1]}_{int(m[2]) + 1}"
      assert retstr not in self._names # sanity check

    # update to the latest renaming of this name string
    self._names[nmstr]  = retstr
    self._ctxt[nm]      = retstr
    return retstr

  def shape_str(self, shape):
    return tuple( str(r) if is_pos_int(r) else self._ctxt[r] for r in shape )

  def __call__(self, vs, sizes, relations, out):
    vals_in_order   = []
    vals_by_name    = {}
    # build lookup by name
    for v,vd    in zip(vs, self._ast.vars):
      vals_by_name[vd.name]   = v
    for sz,szd  in zip(szs, self._ast.sizes):
      vals_by_name[szd.name]  = sz
    for r,rd    in zip(rs, self._ast.relations):
      vals_by_name[rd.name]   = r
    # then lookup values in argument order
    for nm in self._ast.arg_order:
      vals_in_order.append( vals_by_name[nm] )
    vals_in_order.append( out )

    return self._jax_fn(*vals_in_order)

  def comp_body(self, e):
    if type(e) is AST.Let:
      stmts = e.stmts.copy()
      e     = e.ret
    else:
      stmts = []
    stmts.append(AST.assign(self._out_sym, e.type, e, e.srcinfo))

    for s in stmts:
      self.comp_stmt(s)
    self._body.append("  return output")

  def comp_stmt(self, stmt):
    def comp_rhs(e):
      """ This sub-routine should return an np array of
          shape `e.type.shape()` """
      assert not e.type.has_tuples()
      self._shape = e.type.shape_or_scalar()

      # peel off Gen operations here, and stash indices...
      out_idx = []
      while type(e) is AST.Gen:
        out_idx.append(e.name)
        e     = e.body
      self._out_idx = tuple(out_idx)
      assert len(out_idx) == len(self._shape)

      return self.comp_e(e)

    def rec_tuples(e):
      if type(e.type) is T.Tuple:
        key   = str(e.type)
        self._locmap['_named_tuples'][key] = get_python_named_tuple(e.type)
        assert type(e) is AST.Tuple
        args  = [ rec_tuples(a) for a in e.args ]
        return f"_named_tuples['{key}']({','.join(args)})"
      else:
        return comp_rhs(e)
    
    rhs = rec_tuples(stmt.rhs)
    self._body.append(f"  {self.new_name(stmt.name)} = {rhs}")

  def comp_e(self, node, no_broadcast=False):
    if type(node) is AST.Var or type(node) is AST.Proj:
      # extract projected variable
      def get_str(e):
        if type(e) is AST.Var:
          return self._ctxt[e.name]
        elif type(e) is AST.Proj:
          return f"{get_str(e.arg)}[{e.idx}]"
        else: assert False, "bad case"
      res     = get_str(node)

      # corner case:  if the variable is un-accessed or only partially
      #               accessed, it might still be the case that
      #               the enclosing context requires it to be
      #               further broadcast...
      if len(self._shape) > 0 and not no_broadcast:
        shape = self._shape + node.type.shape_or_scalar()
        res   = f"np.broadcast_to({res},{self.shape_str(shape)})"

      return res

    elif type(node) is AST.Access:
      assert type(node.base) is AST.Var or type(node.base) is AST.Proj
      assert len(node.base.type.shape_or_scalar()) == len(node.idx)
      # get the input; i.e. the buffer being accessed
      res     = self.comp_e(node.base, no_broadcast=True)

      # figure out how to re-broadcast, shift etc.
      aff     = [ self.simpl_idx_acc(i) for i in node.idx ]
      axs     = [ a.x for a in aff if a.x != None ]

      # first, slice out any constant accesses
      if any( a.x == None for a in aff ):
        idx   = [ (a.c if a.x == None else ':') for a in aff ]
        idx   = [ "0" if c == "" else c for c in idx ]
        res   = f"{res}[{','.join(idx)}]"
        aff   = [ a for a in aff if a.x != None ]

      # if there is any offset shifting of the access
      if any( a.c != "" for a in aff ):
        shift = tuple( f"-({a.c})" for a in aff if a.c != "" )
        axis  = tuple( i for i,a in enumerate(aff) if a.c != "" )
        res   = f"np.roll({res},{shift},axis={axis})"

      # next, plan for any permutation, repetition of indices
      # as well as broadcasting...
      #idx     = self._out_idx
      #shape   = self._shape.copy()
      unused  = [ (i,x) for i,x in enumerate(self._out_idx) if x not in axs ]
      used    = [ (i,x) for i,x in enumerate(self._out_idx) if x in axs ]
      in_to_out = [ i for i,x in unused ] + [ i for i,x in used ]
      out_to_in = [ i for i in range(len(in_to_out)) ]
      for ii,io in enumerate(in_to_out):
        out_to_in[io] = ii

      # first do the indexing permutation & repetition
      if len(aff) > 0 and (len(aff) != len(used) or
                           any( x != a.x for (i,x),a in zip(used,aff))):
        einnm = { x: chr(97 + i) for i,x in enumerate(used) }
        ein   = ''.join([ einnm[a.x] for a in aff ])
        eout  = ''.join([ einnm[x] for i,x in used ])
        res   = f"np.einsum('{ein}->{eout}',{res})"

      # then do broadcasting and axis shuffling if necessary
      if len(unused) > 0:
        shape         = self.shape_str(self._shape)
        unused_shape  = [ shape[i] for i,_ in unused  ]
        used_shape    = [ shape[i] for i,_ in used    ]
        new_shape     = unused_shape + used_shape
        res   = f"np.broadcast_to({res},{new_shape})"
        res   = f"np.transpose({res},{tuple(out_to_in)})"

      return res

    elif type(node) is AST.Const:
      assert node.type == T.num
      if len(self._shape) > 0 and not no_broadcast:
        return f"np.broadcast_to([{node.val}],{self.shape_str(self._shape)})"
      else:
        return str(node.val)

    elif type(node) is AST.BinOp:
      # do relevant binop?
      pass
    elif type(node) is AST.Sum:
      oldidx    = self._out_idx
      oldshape  = self._shape
      self._out_idx = oldidx + [node.name]
      self._shape   = oldshape + [node.range]

      res       = self.comp_e(node.body)
      res       = f"np.sum({res},axis=-1)"

      self._out_idx = oldidx
      self._shape   = oldshape
      return res

    elif type(node) is AST.BuiltIn:
      # defer
    elif type(node) is AST.Indicate:
      # self._shape
      # what's the best way to do this?
    elif type(node) is AST.Tuple:
      assert False, "Tuple unsupported"
    elif type(node) is AST.TensorLit:
      assert False, "TensorLit unsupported"
    elif type(node) is AST.Let:
      assert False, "Let unsupported"
    elif type(node) is AST.Gen:
      assert False, "Gen unsupported"
    else: assert False, "bad case"


  def simpl_idx_acc(self, i):
    a = self.simpl_idx_acc(i)

    # do not handle full affine indexing for now...
    if len(a.terms) > 1:
      raise NotImplementedError("shear-indexing in JAX unsupported...")
    t = a.terms[0] if len(a.terms) > 0 else None
    if t and t.c != 1:
      raise NotImplementedError("JAX up/down-sample unsupported")
    
    return Aff(a.off, t.x if t else None)


  def simpl_idx(self, i):
    """ returns Aff(off:str,[ATerm(c:int,x:sym)]) """
    def mul_aff(c, aff):
      return Aff( c * aff.off, [ ATerm(c*t.c,t.x) for t in aff.terms ] )
    def rec_sym(a):
      """ recursively convert AST.index to (x,y) where
            x - Aff() type object representing offset
            y - [ATerm] representing sum of scaled index variables """
      if type(a) is AST.IdxConst:
        return Aff(a.val,[]), []
      elif type(a) is AST.IdxVar:
        return [], [ATerm(1,a.name)]
      elif type(a) is AST.IdxSize:
        return Aff(0,[ATerm(1,a.name)]), []
      elif type(a) is AST.IdxAdd or type(a) is AST.IdxSub:
        loff,lx   = rec_sym(a.lhs)
        roff,rx   = rec_sym(a.rhs)
        if type(a) is AST.IdxSub:
          roff    = mul_aff(-1,roff)
          rx      = [ ATerm(-t.c,t.x) for t in rx ]
        return Aff(roff.off + loff.off, roff.terms + loff.terms), (lx + rx)
      elif type(a) is AST.IdxScale:
        off, xs   = rec_sym(a.idx)
        c         = int(a.coeff)
        assert c == a.coeff
        return mul_aff(c, off), [ ATerm(c*t.c,t.x) for t in xs ]

    off, xs = rec_sym(i)


    # having processed a nested expression down into this list form
    # we now want to de-duplicate variable occurrences, whether of size
    # variables or of index variables
    def simpl_aterms(xs):
      res = []
      for x,ts in itertools.groupby(xs, lambda t: t.x)
        c = math.prod([ c for c,_ in ts ])
        if c != 0:
          res.append( ATerm(c,x) )
      return res

    off.terms = simpl_aterms(off.terms)
    xs        = simpl_aterms(xs)

    def str_aff(aff):
      ret = ""
      if aff.off != 0:
        ret = str(aff.off)
      for c,x in aff.terms:
        t = self._ctxt[x]
        if c < 0:
          c   = -c
          neg = True
        if c != 1:
          t = f"{c}*{t}"

        if ret == "":
          if neg:
            ret = "-"+t
          else:
            ret = t
        else:
          if neg:
            ret += " - " + t
          else:
            ret += " + " + t
      return ret

    return Aff( str_aff(off), xs )











