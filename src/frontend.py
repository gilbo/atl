

from adt import ADT
from adt import memo as ADTmemo

from prelude import *

import atl_types as T

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

UST = ADT("""
module Untyped_AST {
  function  = ( name?     name,
                var_decl* vars,
                sym*      sizes,
                rel_decl* relations
                expr      body )

  var_decl  = ( name name, type   type  )
  rel_decl  = ( name name, range* sizes )

  expr = Var      ( name  name )
       | Const    ( float val  )
       | Add      ( expr lhs, expr rhs )
       | Mul      ( expr lhs, expr rhs )
       | Pair     ( expr lhs, expr rhs )
       | Proj     ( int01 idx, expr arg )
       | Gen      ( name name, range range, expr body )
       | Sum      ( name name, range range, expr body )
       | Access   ( expr    base,  index* idx )
       -- implied multiplication of the bracket with body
       | Indicate ( pred    pred,  expr body )
       -- important to express sharing of computation results
       | Let      ( assign* stmts, expr ret )

  assign  = ( name name, type? type, expr rhs )
  
  index   = IdxConst  ( int      val  )
          | IdxVar    ( name     name )
          | IdxSize   ( sym      name )
          | IdxAdd    ( index    lhs,   index rhs )
          | IdxScale  ( fraction coeff, index idx )
  
  pred    = Cmp       ( pred_op op, index lhs,  index rhs )
          | Relation  ( name name,  index* args )
          | Conj      ( pred lhs,   pred rhs    )
          | Disj      ( pred lhs,   pred rhs    )
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'int01':    lambda x: x == 0 or x == 1,
})

# Typed AST
AST = ADT("""
module AST {
  function  = ( name?     name,
                var_decl* vars,
                sym*      sizes,
                rel_decl* relations
                expr      body )

  var_decl  = ( sym name, type   type  )
  rel_decl  = ( sym name, range* sizes )

  expr = Var      ( sym   name )
       | Const    ( float val  )
       | Add      ( expr lhs, expr rhs )
       | Mul      ( expr lhs, expr rhs )
       | Pair     ( expr lhs, expr rhs )
       | Proj     ( int01 idx, expr arg )
       | Gen      ( sym name, range range, expr body )
       | Sum      ( sym name, range range, expr body )
       | Access   ( expr    base,  index* idx )
       -- implied multiplication of the bracket with body
       | Indicate ( pred    pred,  expr body )
       -- important to express sharing of computation results
       | Let      ( assign* stmts, expr ret )
       attributes( type type )

  assign  = ( sym name, type type, expr rhs )
  
  index   = IdxConst  ( int      val  )
          | IdxVar    ( sym      name )
          | IdxSize   ( sym      name )
          | IdxAdd    ( index    lhs,   index rhs )
          | IdxScale  ( fraction coeff, index idx )
  
  pred    = Cmp       ( pred_op op, index lhs,  index rhs )
          | Relation  ( sym name,   index* args )
          | Conj      ( pred lhs,   pred rhs    )
          | Disj      ( pred lhs,   pred rhs    )
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'int01':    lambda x: x == 0 or x == 1,
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
def _expr_str(e,prec=0):
  return str(e.name)
@extclass(UST.Const)
@extclass(AST.Const)
def _expr_str(e,prec=0):
  return str(e.val)
@extclass(UST.Add)
@extclass(AST.Add)
def _expr_str(e,prec=0):
  s = f"{e.lhs._expr_str(50)} + {e.rhs._expr_str(51)}"
  return f"({s})" if prec > 50 else s
@extclass(UST.Mul)
@extclass(AST.Mul)
def _expr_str(e,prec=0):
  s = f"{e.lhs._expr_str(60)} * {e.rhs._expr_str(61)}"
  return f"({s})" if prec > 60 else s
@extclass(UST.Pair)
@extclass(AST.Pair)
def _expr_str(e,prec=0):
  return f"({e.lhs._expr_str(0)},{e.rhs._expr_str(0)})"
@extclass(UST.Proj)
@extclass(AST.Proj)
def _expr_str(e,prec=0):
  return f"({e.arg._expr_str(70)}.{e.idx})"
@extclass(UST.Gen)
@extclass(UST.Sum)
@extclass(AST.Gen)
@extclass(AST.Sum)
def _expr_str(e,prec=0):
  op = "Sum" if (type(e) is AST.Sum or type(e) is UST.Sum) else "Gen"
  s = f"{op}[{str(e.name)}:{e.range}] {e.body._expr_str(10)}"
  return f"({s})" if prec > 10 else s
@extclass(UST.Access)
@extclass(AST.Access)
def _expr_str(e,prec=0):
  idx = ",".join([ str(i) for i in e.idx ])
  s = f"{e.base._expr_str(80)}[{idx}]"
  return f"({s})" if prec > 80 else s
@extclass(UST.Indicate)
@extclass(AST.Indicate)
def _expr_str(e,prec=0):
  s = f"[{e.pred}]*{e.body._expr_str(61)}"
  return f"({s})" if prec > 60 else s
@extclass(UST.Let)
@extclass(AST.Let)
def _expr_str(e,prec=0):
  # note that this is ill-behaved formatting
  # for lets nested inside of expressions
  stmts = [ (f"{str(s.name)}"
             f"{'' if s.type is None else ' : '+str(s.type)} = "
             f"{s.rhs._expr_str(0)} in")
            for s in e.stmts ]
  stmts = "let " + ("\n    ".join(stmts))
  s = f"{stmts}\n    {e.body._expr_str(0)}"
  if prec > 0: s = f"({s})"

del _expr_str
AST.expr.__str__ = lambda e: e._expr_str()
UST.expr.__str__ = lambda e: e._expr_str()

#     String Representations for Index Expressions     #

@extclass(UST.IdxConst)
@extclass(AST.IdxConst)
def _index_str(e,prec=0):
  return str(e.val)
@extclass(UST.IdxVar)
@extclass(UST.IdxSize)
@extclass(AST.IdxVar)
@extclass(AST.IdxSize)
def _index_str(e,prec=0):
  return str(e.name)
@extclass(UST.IdxAdd)
@extclass(AST.IdxAdd)
def _index_str(e,prec=0):
  s = f"{e.lhs._index_str(50)} + {e.rhs._index_str(51)}"
  return f"({s})" if prec > 50 else s
@extclass(UST.IdxScale)
@extclass(AST.IdxScale)
def _index_str(e,prec=0):
  s = f"{e.coeff} * {e.rhs._index_str(61)}"
  return f"({s})" if prec > 60 else s

del _index_str
AST.index.__str__ = lambda e: e._index_str()
UST.index.__str__ = lambda e: e._index_str()

#     String Representations for Predicates     #

@extclass(UST.Cmp)
@extclass(AST.Cmp)
def _pred_str(p,prec=0):
  op_prec = _AST_op_prec[p.op]
  s = f"{e.lhs._pred_str(op_prec)} {p.op} {e.rhs._pred_str(op_prec+1)}"
  return f"({s})" if prec > op_prec else s
@extclass(UST.Cmp)
@extclass(AST.Cmp)
def _pred_str(p,prec=0):
  args = ",".join([ str(i) for i in p.args ])
  s = f"{str(p.name)}[{idx}]"
  return f"({s})" if prec > 80 else s
@extclass(UST.Conj)
@extclass(UST.Disj)
@extclass(AST.Conj)
@extclass(AST.Disj)
def _pred_str(p,prec=0):
  op = "and" if pclass is AST.Conj else "or"
  op_prec = _AST_op_prec[p.op]
  s = f"{e.lhs._pred_str(op_prec)} {p.op} {e.rhs._pred_str(op_prec+1)}"
  return f"({s})" if prec > op_prec else s

del _pred_str
AST.pred.__str__ = lambda p: p._pred_str()
UST.pred.__str__ = lambda p: p._pred_str()


@extclass(UST.function)
@extclass(AST.function)
def _function_str(f):
  sstr  = "sizes    "+(', '.join([ str(s) for s in f.sizes ]))
  vstr  = "vars     "+(', '.join([ f"{str(vd.name)}:{vd.type}"
                                  for vd in f.vars ]))
  rstr  = "rels     "+(', '.join([
      str(rd.name)+'('+(','.join([str(s) for s in rd.sizes]))+')'
      for rd in f.relations ]))
  bstr  = f"return   {f.body}"
  nmstr = "" if f.name is None else f.name

  return f"function {nmstr}\n{sstr}\n{vstr}\n{rstr}\n{bstr}"
del _function_str
AST.function.__str__ = lambda f: f._function_str()
UST.function.__str__ = lambda f: f._function_str()


# def IR_latex_str(e,prec=0):
#     eclass = type(e)
#     s      = "ERROR"
#     if   eclass is IR.Var:
#         s = e.name
#     elif eclass is IR.Const:
#         s = str(e.val)
#     elif eclass is IR.Add:
#         s = f"{IR_latex_str(e.lhs,2)} + {IR_latex_str(e.rhs,2)}"
#         if prec > 2: s = f"\\left({s}\\right)"
#     elif eclass is IR.Mul:
#         s = f"{IR_latex_str(e.lhs,3)} \\cdot {IR_latex_str(e.rhs,3)}"
#         if prec > 3: s = f"\\left({s}\\right)"
#     elif eclass is IR.Pair:
#         s = f"\\left({IR_latex_str(e.lhs,0)},{IR_latex_str(e.rhs,0)}\\right)"
#     elif eclass is IR.Proj:
#         s = f"\\pi_{{{e.idx}}} {IR_latex_str(e.arg,4)}"
#         if prec > 4: s = f"\\left({s}\\right)"
#     elif eclass is IR.Gen or eclass is IR.Sum:
#         op = "\\sum" if eclass is IR.Sum else "\\boxplus"
#         s = f"{op}_{{{e.idxname}:{e.range}}}\\ {IR_latex_str(e.body,1)}"
#         if prec > 1: s = f"\\left({s}\\right)"
#     elif eclass is IR.Access:
#         s = f"{IR_latex_str(e.base,5)}[{e.idx}]"
#         if prec > 5: s = f"\\left({s}\\right)"
#     elif eclass is IR.Indicate:
#         assert isinstance(e.arg, IR.Eq), 'sanity: pred is Eq'
#         s = f"[{e.arg.lhs}={e.arg.rhs}]\\cdot {IR_latex_str(e.body,3)}"
#         if prec > 3: s = f"\\left({s}\\right)"
#     elif eclass is IR.Let:
#         # note that this is ill-behaved formatting
#         # for lets nested inside of expressions
#         s = (f"\\begin{{array}}{{l}}"
#              f" \\textrm{{let }} {e.name} = "
#              f"{IR_latex_str(e.rhs,0)}\\textrm{{ in}}\\\\"
#              f" {IR_latex_str(e.body,0)}"
#              f"\\end{{array}}")
#         if prec > 0: s = f"\\left({s}\\right)"
#     return s
# 
# def IR_latex_repr(e):
#     return f"${IR_latex_str(e)}$"
# 
# IR.expr._repr_latex_ = IR_latex_repr



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# (Basic) Type-Checking

class TCError(Exception):
  def __init__(self, errs):
    errmsg = ("errors during typechecking:\n" +
              ('\n'.join(errs)))
    super(TCError, self).__init__(errmsg)

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

    # var, size, rel definitions
    # additionally create the newly symbol-ified output AST header
    vs, szs, rels     = [], [], []
    for v in func.vars:
      if self._ctxt.get(v.name) is None:
        V = _Var(v.name,v.type)
        self._ctxt.set(v.name,V)
        vs.append( AST.var_decl(V.sym, V.type) )
      else:
        self._err(f"variable name '{v.name}' already used")
    for nm in func.sizes:
      if self._ctxt.get(nm) is None:
        S = _Size(nm)
        self._ctxt.set(nm,S)
        szs.append( S.sym )
      else:
        self._err(f"size variable name '{nm}' already used")
    for r in func.relations:
      if self._ctxt.get(r.name) is None:
        sizes   = [ self._get_range(r,s) for s in r.sizes ]
        R = _Rel(r.name,sizes)
        self._ctxt.set(r.name,R)
        rels.append( AST.rel_decl(R.sym, sizes) )
      else:
        self._err(f"relation name '{r.name}' already used")
    # if the header had name clashes, then early error exit
    # and don't even bother checking the body of the function
    self._report_errors()

    # with the new symbols bound, type the body of the function
    self._ctxt.push()
    body              = self.check(func.body)
    self._ctxt.pop()

    # finally, cache the typed AST on this pass object
    # and report errors encountered while type-checking the body
    self._out_func    = AST.function(func.name, vs, szs, rels, body)
    self._report_errors()

  def typed_ast(self):
    """ Extract the resulting AST """
    return self._out_func
  
  def _err(self, node, msg):
    # might want to discern location
    # via `node` eventually
    self._errors.append(msg)
  
  def _report_errors(self):
    """ Check for and raise accumulated type-checking errors
        together, if any were found """
    if len(self._errors) > 0:
      raise TCError('Found errors during typechecking:\n  '+
                    '\n  '.join(self._errors))
  
  def _get_var(self, node, name):
    """ Retrieve a variable and error if a non-variable was found """
    V = self._ctxt.get(name)
    if V == None:
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
    if R == None:
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
    if S == None:
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
    if I == None:
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
      return AST.Var( nm, typ )

    elif nclass is UST.Const:
      return AST.Const( node.val, T.num )
    
    elif nclass is UST.Add or nclass is UST.Mul:
      lhs   = self.check(node.lhs)
      rhs   = self.check(node.rhs)
      typ   = lhs.type
      if lhs.type == T.error or rhs.type == T.error:
        typ = T.error
      elif lhs.type != rhs.type:
        op  = '+' if nclass is UST.Add else '*'
        self._err(node, f"expected types of operands to match"
                        f"for binary operator '{op}': {node}")
        typ = T.error
      elif nclass is UST.Mul and lhs.type != T.num:
        self._err(node, f"expected scalar number operands to '*': {node}")
        typ = T.error

      cstr  = AST.Add if nclass is UST.Add else AST.Mul
      return cstr( lhs, rhs, typ )
    
    elif nclass is UST.Pair:
      lhs   = self.check(node.lhs)
      rhs   = self.check(node.rhs)
      typ   = T.Pair(lhs.type, rhs.type)
      if lhs.type == T.error or rhs.type == T.error:
        typ = T.error
      return AST.Pair( lhs, rhs, typ )
    
    elif nclass is UST.Proj:
      arg   = self.check(node.arg)
      typ   = T.error
      if arg.type == T.error: pass
      elif type(arg.type) is not T.Pair:
        self._err(node, f"Was expecting a pair as argument: {node}")
      elif node.idx == 0: typ = arg.type.fst
      else:               typ = arg.type.snd
      return AST.Proj(node.idx, arg, typ)
    
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
        return T.Tensor(rng, body.type)
      bigop = AST.Gen if nclass is UST.Gen else AST.Sum
      return bigop(I.sym, rng, body, typ)
    
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
      return AST.Access(base, idx, typ)
    
    elif nclass is UST.Indicate:
      pred    = self.check(node.pred)
      body    = self.check(node.body)
      return AST.Indicate(pred, body, body.type)
    
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
        stmts.append( AST.assign(V.name, V.type, rhs) )

      # and then the return expression
      ret     = self.check(node.ret)
      self._ctxt.pop()

      return AST.Let(stmts,ret, ret.type)

    #          index expressions          #

    elif nclass is UST.IdxConst:
      return AST.IdxConst(node.val)

    elif nclass is UST.IdxVar:
      nm, rng = self._get_index(node,node.name)
      if nm is None:
        return AST.IdxVar( Sym("Error") )
      else:
        return AST.IdxVar( nm )

    elif nclass is UST.IdxSize:
      nm = self._get_range(node,node.name)
      if nm is None:
        return AST.IdxSize( Sym("Error") )
      else:
        return AST.IdxSize( nm )

    elif nclass is UST.IdxAdd:
      lhs     = self.check(node.lhs)
      rhs     = self.check(node.rhs)
      return AST.IdxAdd(lhs, rhs)

    elif nclass is UST.IdxScale:
      idx     = self.check(node.idx)
      return AST.IdxScale(node.coeff, idx)

    #          predicate expressions          #

    elif nclass is UST.Cmp:
      lhs     = self.check(node.lhs)
      rhs     = self.check(node.rhs)
      return AST.Cmp(node.op, lhs, rhs)

    elif nclass is UST.Relation:
      args    = [ self.check(i) for i in node.args ]
      nm, szs = self._get_rel(node,node.name)
      if nm is None: nm = Sym("Error")
      elif len(szs) != len(args):
        self._err(node, f"expected {len(szs)} arguments to relation "
                        f"'{nm}', but got {len(args)}")
      return AST.Relation(nm,args)

    elif nclass is UST.Conj or nclass is UST.Disj:
      lhs     = self.check(node.lhs)
      rhs     = self.check(node.rhs)
      p_op    = AST.Conj if nclass is UST.Conj else AST.Disj
      return p_op(lhs, rhs)

    #          catch-all error          #

    else:
      assert false, "Unexpected UST class for {node}"

def _UST_function_typecheck(f):
  return _TypeChecker(f).typed_ast()
UST.function.typecheck = _UST_function_typecheck
del _UST_function_typecheck


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #



