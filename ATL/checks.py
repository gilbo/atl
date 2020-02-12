
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from . import builtins as B
from .frontend import AST

from fractions import Fraction

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

SZ = {}

class TC_Lite:
  """ A typechecking pass for internal use only """
  def __init__(self, f):
    self._ctxt    = Context()
    self._func    = f
    self._errors  = []

    # pack context
    for sz in f.sizes:
      if self._ctxt.get(sz.name):
        self._err(sz, f"size name {sz.name} already in use")
      self._ctxt.set(sz.name, SZ)
    for vd in f.vars:
      if self._ctxt.get(vd.name):
        self._err(vd, f"var name {vd.name} already in use")
      self._ctxt.set(vd.name, vd.type)
    for rd in f.relations:
      if self._ctxt.get(rd.name):
        self._err(rd, f"rel name {rd.name} already in use")
      self._ctxt.set(rd.name, rd.sizes)

    # check argument order for consistency...
    for nm in f.arg_order:
      if not self._ctxt.get(nm):
        self._err(f, f"arg {nm} in arg_order but not other lists")
    if len(f.arg_order) != len(f.sizes) + len(f.vars) + len(f.relations):
      self._err(f, "fewer args in arg_order than in other lists")

    self._report_errors()

    self._ctxt.push()
    bodytype      = self.check(f.body)
    self._ctxt.pop()

    self._report_errors()

  def _err(self, node, msg):
    self._errors.append(f"{node.srcinfo}: {msg}")

  def _report_errors(self):
    if len(self._errors) > 0:
      fn          = str(self._func)
      es          = '\n'.join(self._errors)
      errmsg      = (f"function: {fn}\n** errors during typechecking:\n{es}")

  def check(self,e):
    eclass = type(e)

    if eclass is AST.Var:
      nm, typ     = e.name, self._ctxt.get(e.name)
      if not T.is_type(typ):
        self._err(e,f"var name '{nm}' undefined")
        typ       = T.error
      elif typ != T.error:
        self._err(e,f"mismatched type with node")
      return e.type

    elif eclass is AST.Const:
      assert e.type == T.num
      return T.num

    elif eclass is AST.BinOp:
      ltyp        = self.check(e.lhs)
      rtyp        = self.check(e.rhs)
      typ         = T.error
      if ltyp is T.error or rtyp is T.error:
        pass
      elif ltyp != rtyp:
        self._err(e, f"expected types of operands to '{e.op}' to match")
      elif e.op != "+" and ltyp != T.num:
        self._err(e, f"expected scalar operands to '{e.op}'")
      elif ltyp != e.type:
        self._err(e,f"mismatched type with node")
      else:
        typ = ltyp
      return e.type

    elif eclass is AST.Tuple:
      atyps       = [ self.check(a) for a in e.args ]
      typ         = T.Tuple(None, atyps)
      for a in atyps:
        if a is T.error:
          typ     = T.error
      if typ != T.error and not typ.matches(e.type):
        self._err(e,f"mismatched type with node")
      return e.type

    elif eclass is AST.Proj:
      atyp        = self.check(e.arg)
      typ         = T.error
      if atyp is T.error: pass
      elif type(atyp) is not T.Tuple:
        self._err(e, "was expecting a tuple argument to Proj")
      elif e.idx >= len(atyp.types):
        self._err(e, f"cannot project entry {e.idx} of an "
                     f"{len(atyp.types)}-tuple")
      else:
        typ       = atyp.types[e.idx]
        if typ != T.error and typ != e.type:
          self._err(e,f"mismatched type with node")
      return e.type

    elif eclass is AST.TensorLit:
      atyps       = [ self.check(a) for a in e.args ]
      typ         = atyps[0]
      for i,a in enumerate(atyps[1:]):
        if typ is not T.error and a is not T.error:
          if typ != a:
            self._err(e, f"expected all entries to have the same "
                         f"type ({typ}), but argument {i+1} had type "
                         f"{t}")
      if typ != T.error:
        typ       = T.Tensor(len(atyps), typ)
        if typ != e.type:
          self._err(e,"mismatched type with node")
      return e.type

    elif eclass is AST.Gen or eclass is AST.Sum:
      if (not is_pos_int(e.range) and
          self._ctxt.get(e.range) != SZ):
        self._ctxt(e,"could not find size variable '{e.range}'")

      self._ctxt.push()
      self._ctxt.set(e.name, e.range)
      btyp        = self.check(e.body)
      self._ctxt.pop()

      typ         = T.Tensor(e.range, btyp) if eclass is AST.Gen else btyp
      if typ != T.error and typ != e.type:
        self._err(e,"mismatched type with node")
      return e.type

    elif eclass is AST.Access:
      typ         = self.check(e.base)
      for i in e.idx:
        self.check_idx(i)
      if typ is T.error: pass
      else:
        for k,i in enumerate(e.idx):
          if type(typ) != T.Tensor:
            self._err(e, f"was expecting {len(idx)} tensor dimensions "
                         f"to index, but found {k}")
            typ   = T.error
            break
          else:
            typ   = typ.type
      if typ != T.error and typ != e.type:
        self._err(e,"mismatched type with node")
      return e.type

    elif eclass is AST.BuiltIn:
      atyps       = [ self.check(a) for a in e.args ]
      typ         = T.error
      try:
        typ       = e.f.typecheck(*atyps)
      except B.BuiltIn_Typecheck_Error as be:
        self._err(e, str(be))
        typ       = T.error
      if typ != T.error and typ != e.type:
        self._err(e,"mismatched type with node")
      return e.type

    elif eclass is AST.Indicate:
      self.check_pred(e.pred)
      typ         = self.check(e.body)
      if typ != T.error and typ != e.type:
        self._err(e,"mismatched type with node")
      return e.type

    elif eclass is AST.Let:
      self._ctxt.push()
      for s in e.stmts:
        rtyp      = self.check(s.rhs)
        if rtyp != T.error and rtyp != s.type:
          self._err(e,"mismatched type with stmt type")
        self._ctxt.set(s.name, s.type)

      # final return expr
      typ         = self.check(e.ret)
      if typ != T.error and typ != e.type:
        self._err(e,"mismatched type with node")
      return e.type
    else: assert False, "unexpected case"

  def check_idx(self,i):
    iclass        = type(i)
    if iclass is AST.IdxConst: pass
    elif iclass is AST.IdxVar:
      nm, rng     = i.name, self._ctxt.get(i.name)
      if not rng:
        self._err(i,f"could not find index '{nm}'")
      elif not is_pos_int(rng):
        if type(rng) is not Sym:
          self._err(i,f"expected '{nm}' to have an int or Sym range")
        elif self._ctxt.get(rng) != SZ:
          self._err(i,f"range '{rng}' of index '{nm}' not a size")
    elif iclass is AST.IdxSize:
      if self._ctxt.get(i.name) != SZ:
        self._err(i,f"could not find size variable '{i.name}'")
    elif iclass is AST.IdxAdd or iclass is AST.IdxSub:
      self.check_idx(i.lhs)
      self.check_idx(i.rhs)
    elif iclass is AST.IdxScale:
      self.check_idx(i.idx)
    else: assert False, "unexpected case"

  def check_pred(self,p):
    pclass        = type(p)
    if pclass is AST.Cmp:
      self.check_idx(p.lhs)
      self.check_idx(p.rhs)
    elif pclass is AST.Relation:
      for i in p.args:
        self.check_idx(i)
      szs         = self._ctxt.get(p.name)
      if type(szs) != list:
        self._err(p,f"could not find relation variable '{p.name}'")
      elif len(szs) != len(p.args):
        self._err(p,f"called {len(szs)}-ary relation with {len(szs)} args")
    elif pclass is AST.Conj or pclass is AST.Disj:
      self.check_pred(p.lhs)
      self.check_pred(p.rhs)
    else: assert False, "unexpected case"





# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #






