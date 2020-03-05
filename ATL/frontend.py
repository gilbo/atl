

from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T

import itertools
from fractions import Fraction

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Front-End ASTs

pred_ops = {
  "<":    True,
  ">":    True,
  "<=":   True,
  ">=":   True,
  "==":   True,
}

bin_ops = {
  "+":    True,
  "-":    True,
  "*":    True,
  "/":    True,
}

UST = ADT("""
module Untyped_AST {
  function  = ( name?       name,
                name*       arg_order,
                type?       rettype,
                var_decl*   vars,
                size_decl*  sizes,
                rel_decl*   relations,
                expr        body,
                srcinfo     srcinfo )

  var_decl  = ( name name, type   type,  srcinfo srcinfo )
  size_decl = ( sym  name,               srcinfo srcinfo )
  rel_decl  = ( name name, range* sizes, srcinfo srcinfo )

  expr = Var      ( name  name )
       | Const    ( float val  )
       | BinOp    ( bin_op op, expr lhs, expr rhs )
       | Tuple    ( expr* args )
       | NamedTuple( name* labels, expr* args )
       | Proj     ( plabel idx, expr arg )
       | TensorLit( expr* args )
       | Gen      ( name name, range range, expr body )
       | Sum      ( name name, range range, expr body )
       | Access   ( expr base, index* idx )
       | BuiltIn  ( builtin f, expr* args )
       -- important for some front-end variants and metaprogramming
       -- strategies; always inlined for simplicity for now
       | FuncCall ( typed_ast ast, exprrngrel* args )
       -- implied multiplication of the bracket with body
       | Indicate ( pred pred, expr body  )
       -- important to express sharing of computation results
       | Let      ( assign* stmts, expr ret )
       attributes( srcinfo srcinfo )

  assign  = ( name name, type? type, expr rhs, srcinfo srcinfo )
  
  index   = IdxConst  ( int      val  )
          | IdxVar    ( name     name )
          | IdxSize   ( sym      name )
          | IdxAdd    ( index    lhs,   index rhs )
          | IdxScale  ( fraction coeff, index idx )
          attributes( srcinfo srcinfo )
  
  pred    = Cmp       ( pred_op op, index lhs,  index rhs )
          | Relation  ( name name,  index* args )
          | Conj      ( pred lhs,   pred rhs    )
          | Disj      ( pred lhs,   pred rhs    )
          attributes( srcinfo srcinfo )
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'builtin':  lambda x: isinstance(x, B.BuiltIn),
  'typed_ast':lambda x: type(x) is AST.function,
  'exprrngrel': lambda x: (isinstance(x, UST.expr) or is_valid_name(x)
                           or is_pos_int(x) or type(x) is Sym),
  'pred_op':  lambda x: x in pred_ops,
  'bin_op':   lambda x: x in bin_ops,
  'plabel':   lambda x: type(x) is int or is_valid_name(x),
  'srcinfo':  lambda x: type(x) is SrcInfo,
})

# Typed AST
AST = ADT("""
module AST {
  function  = ( name?       name,
                sym*        arg_order,
                type        rettype,
                var_decl*   vars,
                size_decl*  sizes,
                rel_decl*   relations,
                expr        body,
                srcinfo     srcinfo )

  var_decl  = ( sym name, type   type,  srcinfo srcinfo )
  size_decl = ( sym name,               srcinfo srcinfo )
  rel_decl  = ( sym name, range* sizes, srcinfo srcinfo )

  expr = Var      ( sym   name )
       | Const    ( float val  )
       | BinOp    ( bin_op op, expr lhs, expr rhs )
       | Tuple    ( expr* args )
       | Proj     ( plabel idx, expr arg )
       | TensorLit( expr* args )
       | Gen      ( sym name, range range, expr body )
       | Sum      ( sym name, range range, expr body )
       | Access   ( expr base, index* idx )
       | BuiltIn  ( builtin f, expr* args )
       -- implied multiplication of the bracket with body
       | Indicate ( pred pred, expr body  )
       -- important to express sharing of computation results
       | Let      ( assign* stmts, expr ret )
       attributes( type type, srcinfo srcinfo )

  assign  = ( sym name, type type, expr rhs, srcinfo srcinfo )
  
  index   = IdxConst  ( int      val  )
          | IdxVar    ( sym      name )
          | IdxSize   ( sym      name )
          | IdxAdd    ( index    lhs,   index rhs )
          | IdxSub    ( index    lhs,   index rhs )
          | IdxScale  ( fraction coeff, index idx )
          attributes( srcinfo srcinfo )
  
  pred    = Cmp       ( pred_op op, index lhs,  index rhs )
          | Relation  ( sym name,   index* args )
          | Conj      ( pred lhs,   pred rhs    )
          | Disj      ( pred lhs,   pred rhs    )
          attributes( srcinfo srcinfo )
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'builtin':  lambda x: isinstance(x, B.BuiltIn),
  'pred_op':  lambda x: x in pred_ops,
  'bin_op':   lambda x: x in bin_ops,
  'plabel':   lambda x: type(x) is int,
  'srcinfo':  lambda x: type(x) is SrcInfo,
})


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Display as string

_AST_op_prec = {
  # access      80
  # projection  70
  "*":    60,
  "/":    60,
  "+":    50,
  "-":    50,
  "<":    40,
  ">":    40,
  "<=":   40,
  ">=":   40,
  "==":   30,
  "and":  20,
  "or":   20,
}


#     String Representations for Expressions     #

@extclass(UST.Var)
@extclass(AST.Var)
def _expr_str(e,prec=0,ind=''):
  return repr(e.name)
@extclass(UST.Const)
@extclass(AST.Const)
def _expr_str(e,prec=0,ind=''):
  return str(e.val)
@extclass(UST.BinOp)
@extclass(AST.BinOp)
def _expr_str(e,prec=0,ind=''):
  op_prec = _AST_op_prec[e.op]
  s = (f"{e.lhs._expr_str(op_prec,ind)} {e.op} "
       f"{e.rhs._expr_str(op_prec+1,ind)}")
  return f"({s})" if prec > op_prec else s
@extclass(UST.Tuple)
@extclass(AST.Tuple)
def _expr_str(e,prec=0,ind=''):
  args = ",".join([ a._expr_str(0,ind) for a in e.args ])
  return f"({args})"
@extclass(UST.NamedTuple)
def _expr_str(e,prec=0,ind=''):
  args = ",".join([ f"{l}:{a._expr_str(0,ind)}"
                    for l,a in zip(e.labels,e.args) ])
  return f"{{{args}}}"
@extclass(UST.Proj)
@extclass(AST.Proj)
def _expr_str(e,prec=0,ind=''):
  return f"({e.arg._expr_str(70,ind)}.{e.idx})"
@extclass(UST.TensorLit)
@extclass(AST.TensorLit)
def _expr_str(e,prec=0,ind=''):
  args = ",".join([ a._expr_str(0,ind) for a in e.args ])
  return f"[{args}]"
@extclass(UST.Gen)
@extclass(UST.Sum)
@extclass(AST.Gen)
@extclass(AST.Sum)
def _expr_str(e,prec=0,ind=''):
  op = "Sum" if (type(e) is AST.Sum or type(e) is UST.Sum) else "Gen"
  s = f"{op}[{repr(e.name)}:{e.range}] {e.body._expr_str(10,ind)}"
  return f"({s})" if prec > 10 else s
@extclass(UST.Access)
@extclass(AST.Access)
def _expr_str(e,prec=0,ind=''):
  idx = ",".join([ i._index_str(0,ind) for i in e.idx ])
  s = f"{e.base._expr_str(80,ind)}[{idx}]"
  return f"({s})" if prec > 80 else s
@extclass(UST.BuiltIn)
@extclass(AST.BuiltIn)
def _expr_str(e,prec=0,ind=''):
  args = ",".join([ a._expr_str(0,ind) for a in e.args ])
  s = f"{e.f.name()}({args})"
  return f"({s})" if prec > 80 else s
@extclass(UST.FuncCall)
def _expr_str(e,prec=0,ind=''):
  args = ",".join([ a._expr_str(0,ind) for a in e.args ])
  s = f"{e.ast.name}({args})"
  return f"({s})" if prec > 80 else s
@extclass(UST.Indicate)
@extclass(AST.Indicate)
def _expr_str(e,prec=0,ind=''):
  s = f"[{e.pred}]*{e.body._expr_str(61,ind)}"
  return f"({s})" if prec > 60 else s
@extclass(UST.Let)
@extclass(AST.Let)
def _expr_str(e,prec=0,ind=''):
  # note that this is ill-behaved formatting
  # for lets nested inside of expressions
  subind = ind + "    "
  decls  = [ (f"{repr(s.name)}"
              f"{'' if s.type is None else ' : '+str(s.type)}")
             for s in e.stmts ]
  # compute alignment...
  max_len = 0
  for d in decls:
    max_len = max(max_len, len(d))
  stmts = [ (f"{d}{(max_len-len(d))*' '} = "
             f"{s.rhs._expr_str(0,subind)} in")
            for s,d in zip(e.stmts,decls) ]
  stmts = f"let \n{subind}" + (f"\n{subind}".join(stmts))
  s = f"{stmts}\n{subind}{max_len*' '} {e.ret._expr_str(0,subind)}"
  return f"({s})" if prec > 0 else s

del _expr_str
AST.expr.__str__ = lambda e: e._expr_str()
UST.expr.__str__ = lambda e: e._expr_str()

#     String Representations for Index Expressions     #

@extclass(UST.IdxConst)
@extclass(AST.IdxConst)
def _index_str(e,prec=0,ind=''):
  return str(e.val)
@extclass(UST.IdxVar)
@extclass(UST.IdxSize)
@extclass(AST.IdxVar)
@extclass(AST.IdxSize)
def _index_str(e,prec=0,ind=''):
  return repr(e.name)
@extclass(UST.IdxAdd)
@extclass(AST.IdxAdd)
def _index_str(e,prec=0,ind=''):
  s = f"{e.lhs._index_str(50,ind)} + {e.rhs._index_str(51,ind)}"
  return f"({s})" if prec > 50 else s
@extclass(AST.IdxSub)
def _index_str(e,prec=0,ind=''):
  s = f"{e.lhs._index_str(50,ind)} - {e.rhs._index_str(51,ind)}"
  return f"({s})" if prec > 50 else s
@extclass(UST.IdxScale)
@extclass(AST.IdxScale)
def _index_str(e,prec=0,ind=''):
  s = f"{e.coeff} * {e.idx._index_str(61,ind)}"
  return f"({s})" if prec > 60 else s

del _index_str
AST.index.__str__ = lambda e: e._index_str()
UST.index.__str__ = lambda e: e._index_str()

#     String Representations for Predicates     #

@extclass(UST.Cmp)
@extclass(AST.Cmp)
def _pred_str(p,prec=0,ind=''):
  op_prec = _AST_op_prec[p.op]
  s = (f"{p.lhs._index_str(op_prec,ind)} {p.op} "
       f"{p.rhs._index_str(op_prec+1,ind)}")
  return f"({s})" if prec > op_prec else s
@extclass(UST.Relation)
@extclass(AST.Relation)
def _pred_str(p,prec=0,ind=''):
  args = ",".join([ i._index_str(0,ind) for i in p.args ])
  s = f"{str(p.name)}({args})"
  return f"({s})" if prec > 80 else s
@extclass(UST.Conj)
@extclass(UST.Disj)
@extclass(AST.Conj)
@extclass(AST.Disj)
def _pred_str(p,prec=0,ind=''):
  op = "and" if type(p) is AST.Conj else "or"
  op_prec = _AST_op_prec[op]
  s = (f"{p.lhs._pred_str(op_prec,ind)} {op} "
       f"{p.rhs._pred_str(op_prec+1,ind)}")
  return f"({s})" if prec > op_prec else s

del _pred_str
AST.pred.__str__ = lambda p: p._pred_str()
UST.pred.__str__ = lambda p: p._pred_str()


@extclass(UST.function)
@extclass(AST.function)
def __str__(f):
  ind     = "\n         "
  colW    = 80-len(ind)
  sstr    = "sizes    "+ind.join(wrapjoin([ str(sz.name) for sz in f.sizes ],
                                 col_width=colW, delimit=", "))
  vstr    = "vars     "+ind.join(wrapjoin([ f"{str(vd.name)}:{vd.type}"
                                            for vd in f.vars ],
                                          col_width=colW, delimit=", "))
  relnms  = [ str(rd.name) + '(' + ','.join([str(s) for s in rd.sizes]) + ')'
              for rd in f.relations ]
  rstr    = "rels     "+ind.join(wrapjoin(relnms, col_width=colW,
                                                  delimit=", "))
  rtstr  = f"rettype  {f.rettype}"
  bstr   = f"return\n{f.body}"
  nmstr = "" if f.name is None else f.name

  return f"function {nmstr}\n{sstr}\n{vstr}\n{rstr}\n{rtstr}\n{bstr}"
del __str__



# --------------------------------------------------------------------------- #
# AST argument mangling


@extclass(AST.function)
def decls_in_order(self):
  if not hasattr(self,'_decl_cache'):
    def find_sym(nm):
      for d in itertools.chain(self.vars, self.sizes, self.relations):
        if d.name == nm:  return d
      assert False, "Internal inconsistency in typed ast arguments"
    self._decl_cache = [ find_sym(nm) for nm in self.arg_order ]
  return self._decl_cache

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# (Basic) Type-Checking

class TCError(Exception):
  def __init__(self, errs):
    errmsg = ("errors during typechecking:\n" +
              ('\n'.join(errs)))
    super().__init__(errmsg)

class _Var:
  """ Helper object for TypeChecker pass """
  def __init__(self, name, type):
    self.sym  = Sym(name)
    self.type = type

class _IVar:
  """ Helper object for TypeChecker pass """
  def __init__(self, name, range):
    self.sym    = Sym(name)
    self.range  = range

class _Size:
  """ Helper object for TypeChecker pass """
  def __init__(self, name):
    assert(type(name) is Sym)
    self.sym  = name

class _Rel:
  """ Helper object for TypeChecker pass """
  def __init__(self, name, sizes):
    self.sym    = Sym(name)
    self.sizes  = sizes

class _TypeChecker:
  """ TypeChecker pass; expects an untyped function object """

  def __init__(self, func):
    assert type(func) is UST.function, 'expected UST.function'
    self._ctxt        = Context()
    self._input_func  = func
    self._errors      = []

    # build reverse mapping (arg_idx) of arg_order: Index <-> ArgName
    arg_syms          = [ None for _ in func.arg_order ]
    arg_idx           = {}
    for idx,nm in enumerate(func.arg_order):
      if nm in arg_idx:
        raise TypeError(f"Cannot repeat argument names: '{nm}'")
      arg_idx[nm] = idx
    # check that the number of arguments is consistent...
    len_match = ( len(func.arg_order) == ( len(func.vars) +
                                           len(func.sizes) +
                                           len(func.relations) ))
    assert len_match, "Inconsistent number of arg_order arguments"

    # var, size, rel definitions
    # additionally create the newly symbol-ified output AST header
    vs, szs, rels     = [], [], []
    for v in func.vars:
      if self._ctxt.get(v.name) is None:
        V = _Var(v.name,v.type)
        self._ctxt.set(v.name,V)
        vs.append( AST.var_decl(V.sym, V.type, v.srcinfo) )
        arg_syms[arg_idx[str(v.name)]] = V.sym
      else:
        self._err(v,f"variable name '{v.name}' already used")
    for sz in func.sizes:
      if ( self._ctxt.get(sz.name) is None and
           self._ctxt.get(str(sz.name)) is None ):
        S = _Size(sz.name)
        self._ctxt.set(sz.name,S)
        szs.append( AST.size_decl( S.sym, sz.srcinfo ) )
        arg_syms[arg_idx[str(sz.name)]] = S.sym
      else:
        self._err(sz,f"size variable name '{nm}' already used")
    for r in func.relations:
      if self._ctxt.get(r.name) is None:
        sizes   = [ self._get_range(r,s) for s in r.sizes ]
        R = _Rel(r.name,sizes)
        self._ctxt.set(r.name,R)
        rels.append( AST.rel_decl(R.sym, sizes, r.srcinfo) )
        arg_syms[arg_idx[str(r.name)]] = R.sym
      else:
        self._err(r,f"relation name '{r.name}' already used")
    # if the header had name clashes, then early error exit
    # and don't even bother checking the body of the function
    self._report_errors()

    # with the new symbols bound, type the body of the function
    self._ctxt.push()
    body              = self.check(func.body)
    self._ctxt.pop()
    rettype           = body.type
    if func.rettype is not None:
      if not body.type.matches(func.rettype):
        self._err(body,f"expected function to return value of type "
                       f"{func.rettype}, but got {body.type}")
      rettype = func.rettype

    # finally, cache the typed AST on this pass object
    # and report errors encountered while type-checking the body
    self._out_func    = AST.function(func.name, arg_syms, rettype,
                                     vs, szs, rels,
                                     body, func.srcinfo)
    self._report_errors()

  def typed_ast(self):
    """ Extract the resulting AST """
    return self._out_func
  
  def _err(self, node, msg):
    # might want to discern location
    # via `node` eventually
    self._errors.append(f"{node.srcinfo}: {msg}")
  
  def _report_errors(self):
    """ Check for and raise accumulated type-checking errors
        together, if any were found """
    if len(self._errors) > 0:
      raise TCError(self._errors)
  
  def _get_var(self, node, name):
    """ Retrieve a variable and error if a non-variable was found """
    V = self._ctxt.get(name)
    if V is None:
      self._err(node, f"variable '{name}' was undefined")
      return Sym("Error"), T.error
    elif not type(V) is _Var:
      self._err(node, f"'{name}' was defined as "
                      f"something other than a normal variable")
      return Sym("Error"), T.error
    else: return V.sym, V.type

  def _get_rel(self, node, name):
    """ Retrieve a relation variable and error if not found as such
        Returns symbol for the name and list of argument ranges;
        Returns None instead of a list on failed lookup """
    R = self._ctxt.get(name)
    if R is None:
      self._err(node, f"relation '{name}' was undefined")
      return None
    elif not type(R) is _Rel:
      self._err(node, f"'{name}' was defined as "
                      f"something other than a relation")
      return None
    else: return R.sym, R.sizes

  def _get_range(self, node, name):
    """ Retrieve a size variable and error if not found;
        Accepts and returns integer arguments unmodified """
    if type(name) is int: return name
    S = self._ctxt.get(name)
    if S is None:
      self._err(node, f"size variable '{name}' was undefined")
      return None
    elif not type(S) is _Size:
      self._err(node, f"'{name}' was defined as "
                      f"something other than a size")
      return None
    else: return S.sym

  def _get_index(self, node, name):
    """ Retrieve an index variable and error if not found """
    I = self._ctxt.get(name)
    if I is None:
      self._err(node, f"index variable '{name}' was undefined")
      return None
    elif type(I) is _IVar:
      return I.sym, I.range
    else:
      self._err(node, f"'{name}' was defined as "
                      f"something other than an index variable")
      return None
  
  def check(self, node):
    """ type-check expressions, index-expressions and predicates """
    nclass = type(node)
    if   nclass is UST.Var:
      nm, typ   = self._get_var(node, node.name)
      return AST.Var( nm, typ, node.srcinfo )

    elif nclass is UST.Const:
      return AST.Const( node.val, T.num, node.srcinfo )
    
    elif nclass is UST.BinOp:
      lhs   = self.check(node.lhs)
      rhs   = self.check(node.rhs)
      typ   = lhs.type
      if lhs.type is T.error or rhs.type is T.error:
        typ = T.error
      elif lhs.type != rhs.type:
        self._err(node, f"expected types of operands to match"
                        f"for binary operator '{node.op}'")
        typ = T.error
      elif node.op != "+" and lhs.type != T.num:
        self._err(node, f"expected scalar number operands to "
                        f"'{node.op}'")
        typ = T.error

      return AST.BinOp( node.op, lhs, rhs, typ, node.srcinfo )
    
    elif nclass is UST.Tuple:
      args    = [ self.check(a) for a in node.args ]
      typ     = T.Tuple(None, [ a.type for a in args ])
      for a in args:
        if a.type is T.error:
          typ = T.error
      return AST.Tuple( args, typ, node.srcinfo )

    elif nclass is UST.NamedTuple:
      args    = [ self.check(a) for a in node.args ]
      typ     = T.Tuple(T.labels(node.labels), [ a.type for a in args ])
      for a in args:
        if a.type is T.error:
          typ = T.error
      return AST.Tuple( args, typ, node.srcinfo )
    
    elif nclass is UST.Proj:
      arg   = self.check(node.arg)
      idx   = 0 # default for errors
      typ   = T.error
      if arg.type is T.error: pass
      elif type(arg.type) is not T.Tuple:
        self._err(node, f"Was expecting a tuple as argument")
      else:
        # figure out which numeric index this is...
        idx   = node.idx
        n_tup = len(arg.type.types)
        if type(idx) is int:
          if idx < 0 or idx >= n_tup:
            self._err(node, f"index {idx} was not between 0 and {n_tup}")
          else:
            typ = arg.type.types[idx]
        else:
          assert is_valid_name(idx)
          # find this label...
          for k,nm in enumerate(arg.type.labels.names):
            if nm == idx:
              idx = k
              typ = arg.type.types[idx]
          # if we failed to find it...
          if type(idx) is str:
            self._err(node, f"could not find tuple entry label '{idx}'")
            idx = 0
      return AST.Proj(idx, arg, typ, node.srcinfo)

    elif nclass is UST.TensorLit:
      args    = [ self.check(a) for a in node.args ]
      typ     = args[0].type
      for i,a in enumerate(args):
        if typ is not T.error and a.type is not T.error:
          if typ != a.type:
            self._err(node, f"expected all entries to have the same "
                            f"type ({typ}), but argument {i} had type "
                            f"{a.type}")
      if typ is not T.error:
        typ = T.Tensor(len(args), typ)
      return AST.TensorLit( args, typ, node.srcinfo )
    
    elif nclass is UST.Gen or nclass is UST.Sum:
      self._ctxt.push()
      rng   = self._get_range(node,node.range)
      if rng is None: rng = Sym("Error")
      I     = _IVar(node.name,rng)
      self._ctxt.set(node.name,I)
      body  = self.check(node.body)
      self._ctxt.pop()

      typ   = T.error
      if  body.type is T.error: pass
      elif nclass is UST.Sum:
        typ = body.type
      else: # nclass is IR.Gen
        assert nclass is UST.Gen
        typ = T.Tensor(rng, body.type)
      bigop = AST.Gen if nclass is UST.Gen else AST.Sum
      return bigop(I.sym, rng, body, typ, node.srcinfo)
    
    elif nclass is UST.Access:
      base    = self.check(node.base)
      idx     = [ self.check(i) for i in node.idx ]
      typ     = base.type
      if typ is T.error: pass
      else:
        # recurse through the tensor type to check whether
        # it has enough dimensions to index into
        for k,i in enumerate(idx):
          if type(typ) != T.Tensor:
            self._err(node, f"was expecting {len(idx)} tensor dimensions "
                            f"to index, but found {k}")
            typ = T.error
            break
          else:
            typ = typ.type
        # `typ` should now have the resulting type after len(idx) indexings
      return AST.Access(base, idx, typ, node.srcinfo)

    elif nclass is UST.BuiltIn:
      args    = [ self.check(i) for i in node.args ]
      atyps   = [ a.type for a in args ]
      typ     = T.error
      try:
        typ   = node.f.typecheck(*atyps)
      except B.BuiltIn_Typecheck_Error as be:
        self._err(node, str(be))
        typ   = T.error
      return AST.BuiltIn(node.f, args, typ, node.srcinfo)

    elif nclass is UST.FuncCall:
      decls   = node.ast.decls_in_order()

      # convert the arguments into a remapping dictionary
      typ     = node.ast.rettype
      remap   = {}
      binds   = []
      # first, pre-process size substitutions
      for i,(d,a) in enumerate(zip(decls,node.args)):
        if type(d) is AST.size_decl:
          if type(a) is not Sym and not is_pos_int(a):
            self._err(node, f"was expecting argument {i} to be a range. "
                            f"(size or positive integer)")
            typ       = T.error
          else:
            rng       = self._get_range(node,a)
            if rng is None:  typ           = T.error
            else:           remap[d.name]  = rng
      # then, process the rest of arguments
      for i,(d,a) in enumerate(zip(decls,node.args)):
        if type(d) is AST.rel_decl:
          if type(a) is not str:
            self._err(node, f"was expecting argument {i} to be a relation.")
            typ       = T.error
          else:
            nm, szs   = self._get_rel(node,a)
            if nm is None:
              typ     = T.error
            else:
              sizes   = [ (remap[z] if z in remap else z) for z in d.sizes ]
              if tuple(szs) != tuple(sizes):
                self._err(node, f"argument {i} (a relation) did not have "
                                f"matching sizes: {szs} vs {sizes}")
                typ   = T.error
              else:
                remap[d.name] = nm
        elif type(d) is AST.size_decl:
          pass
        elif type(d) is AST.var_decl:
          if not isinstance(a, UST.expr):
            self._err(node, f"was expecting argument {i} to be an expression.")
            typ       = T.error
          else:
            e         = self.check(a)
            if e.type == T.error:
              typ     = T.error
            elif e.type != d.type.remap(remap):
              self._err(a, f"type mismatch; expected {d.type} but got "
                           f"inconsistent {e.type}")
              typ     = T.error
            else:
              nm      = d.name.copy()
              binds.append( AST.assign( nm, d.type.remap(remap),
                                        e, node.srcinfo ) )
              remap[d.name] = nm
      # done processing arguments into a remapping
      # consider early exiting to simplify continuing logic
      if typ == T.error:
        return AST.Const(0.0,T.error,node.srcinfo)

      # ok, now we need to construct an alpha-substituted version of
      # the function as a let-expression
      body = _AlphaSub(node.ast.body, remap).get_expr()
      return AST.Let(binds, body, body.type, node.srcinfo)
    
    elif nclass is UST.Indicate:
      pred    = self.check(node.pred)
      body    = self.check(node.body)
      return AST.Indicate(pred, body, body.type, node.srcinfo)
    
    elif nclass is UST.Let:
      # process all the statements
      stmts   = []
      self._ctxt.push() # allow values to be overwritten...
      for s in node.stmts:
        rhs   = self.check(s.rhs)
        typ   = s.type if s.type != None else rhs.type
        if s.type != None and s.type != rhs.type:
          self._err(s, f"type of right-hand-side ({s.type}) does not "
                       f"match type annotation ({s.type})")
        V     = _Var(s.name,typ)
        self._ctxt.set(s.name,V)
        stmts.append( AST.assign(V.sym, V.type, rhs, s.srcinfo) )

      # and then the return expression
      ret     = self.check(node.ret)
      self._ctxt.pop()

      return AST.Let(stmts,ret, ret.type, node.srcinfo)

    #          index expressions          #

    elif nclass is UST.IdxConst:
      return AST.IdxConst(node.val, node.srcinfo)

    elif nclass is UST.IdxVar:
      nm, rng = self._get_index(node,node.name)
      if nm is None:
        return AST.IdxVar( Sym("Error"), node.srcinfo )
      else:
        return AST.IdxVar( nm, node.srcinfo )

    elif nclass is UST.IdxSize:
      nm = self._get_range(node,node.name)
      if nm is None:
        return AST.IdxSize( Sym("Error"), node.srcinfo )
      else:
        return AST.IdxSize( nm, node.srcinfo )

    elif nclass is UST.IdxAdd:
      lhs     = self.check(node.lhs)
      rhs     = self.check(node.rhs)
      return AST.IdxAdd(lhs, rhs, node.srcinfo)

    elif nclass is UST.IdxScale:
      idx     = self.check(node.idx)
      return AST.IdxScale(node.coeff, idx, node.srcinfo)

    #          predicate expressions          #

    elif nclass is UST.Cmp:
      lhs     = self.check(node.lhs)
      rhs     = self.check(node.rhs)
      return AST.Cmp(node.op, lhs, rhs, node.srcinfo)

    elif nclass is UST.Relation:
      args    = [ self.check(i) for i in node.args ]
      nm, szs = self._get_rel(node,node.name)
      if nm is None: nm = Sym("Error")
      elif len(szs) != len(args):
        self._err(node, f"expected {len(szs)} arguments to relation "
                        f"'{nm}', but got {len(args)}")
      return AST.Relation(nm,args,node.srcinfo)

    elif nclass is UST.Conj or nclass is UST.Disj:
      lhs     = self.check(node.lhs)
      rhs     = self.check(node.rhs)
      p_op    = AST.Conj if nclass is UST.Conj else AST.Disj
      return p_op(lhs, rhs, node.srcinfo)

    #          catch-all error          #

    else:
      assert False, "Unexpected UST class for {node}"

def _UST_function_typecheck(f):
  return _TypeChecker(f).typed_ast()
UST.function.typecheck = _UST_function_typecheck
del _UST_function_typecheck

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class _AlphaSub:
  """ alpha-substitution pass; expects a typed-ast and argument remapping """

  def __init__(self, body, remapping):
    assert isinstance(body, AST.expr), 'expected AST.function'
    assert type(remapping) is dict
    self._ctxt        = Environment()
    for nm,val in remapping.items():
      self._ctxt[nm]  = val
    self._ctxt.push()

    # get the body and sub the expression?
    self._sub_expr = self.alpha(body)

  def get_expr(self):
    return self._sub_expr

  def alpha(self, node):
    nclass  = type(node)
    typ     = ( node.type.remap(self._ctxt)
                  if isinstance(node, AST.expr) else None )

    if   nclass is AST.Var:
      nm    = self._ctxt[node.name]
      return AST.Var( nm, typ, node.srcinfo )
    elif nclass is AST.Const:
      return node
    elif nclass is AST.BinOp:
      lhs   = self.alpha(node.lhs)
      rhs   = self.alpha(node.rhs)
      return AST.BinOp(node.op, lhs, rhs, typ, node.srcinfo)
    elif nclass is AST.Tuple:
      args  = [ self.alpha(a) for a in node.args ]
      return AST.Tuple( args, typ, node.srcinfo )
    elif nclass is AST.Proj:
      arg   = self.alpha(node.arg)
      return AST.Proj(node.idx, arg, typ, node.srcinfo)
    elif nclass is AST.TensorLit:
      args  = [ self.alpha(a) for a in node.args ]
      return AST.TensorLit( args, typ, node.srcinfo )
    elif nclass is AST.Gen or nclass is AST.Sum:
      self._ctxt.push()
      rng   = node.range
      if type(node.range) is Sym and node.range in self._ctxt:
        rng = self._ctxt[node.range]
      nm    = node.name.copy()
      self._ctxt[node.name] = nm
      body  = self.alpha(node.body)
      self._ctxt.pop()
      return nclass(nm, rng, body, typ, node.srcinfo)
    elif nclass is AST.Access:
      base  = self.alpha(node.base)
      idx   = [ self.alpha(i) for i in node.idx ]
      return AST.Access(base, idx, typ, node.srcinfo)
    elif nclass is AST.BuiltIn:
      args  = [ self.alpha(a) for a in node.args ]
      return AST.BuiltIn(node.f, args, typ, node.srcinfo)
    elif nclass is AST.Indicate:
      pred  = self.alpha(node.pred)
      body  = self.alpha(node.body)
      return AST.Indicate(pred, body, typ, node.srcinfo)
    elif nclass is AST.Let:
      stmts = []
      self._ctxt.push() # allow values to be overwritten...
      for s in node.stmts:
        rhs   = self.alpha(s.rhs)
        styp  = s.type.remap(self._ctxt)
        nm    = s.name.copy()
        self._ctxt[s.name] = nm
        stmts.append( AST.assign(nm, styp, rhs, s.srcinfo) )
      ret   = self.alpha(node.ret)
      self._ctxt.pop()
      return AST.Let(stmts, ret, typ, node.srcinfo)

    elif nclass is AST.IdxConst:
      return node
    elif nclass is AST.IdxVar:
      return AST.IdxVar(self._ctxt[node.name], node.srcinfo)
    elif nclass is AST.IdxSize:
      val   = self._ctxt[node.name]
      if type(val) is int:
        return AST.IdxConst(val, node.srcinfo)
      else:
        return AST.IdxSize(val, node.srcinfo)
    elif nclass is AST.IdxAdd or nclass is AST.IdxSub:
      lhs   = self.alpha(node.lhs)
      rhs   = self.alpha(node.rhs)
      return nclass(lhs, rhs, node.srcinfo)
    elif nclass is AST.IdxScale:
      idx   = self.alpha(node.idx)
      return AST.IdxScale(node.coeff, idx, node.srcinfo)

    elif nclass is AST.Cmp:
      lhs   = self.alpha(node.lhs)
      rhs   = self.alpha(node.rhs)
      return AST.Cmp(node.op, lhs, rhs, node.srcinfo)
    elif nclass is AST.Relation:
      args  = [ self.alpha(i) for i in node.args ]
      nm    = self._ctxt[node.name]
      return AST.Relation(nm,args,node.srcinfo)
    elif nclass is AST.Conj or nclass is AST.Disj:
      lhs   = self.alpha(node.lhs)
      rhs   = self.alpha(node.rhs)
      return nclass(lhs, rhs, node.srcinfo)

    else:
      assert False, "Unexpected AST class for {node}"

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

from . import builtins as B 


