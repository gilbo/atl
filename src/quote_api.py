
from prelude import *

import atl_types as T
from frontend import UST, AST
from functions import Function

from py_type_values import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Quoted Code Objects for constructing via external meta-programming

class Var:
  """ Use this class to represent variables """
  def __init__(self,nm):
    self._name    = nm
    self._srcinfo = get_srcinfo(2)
  def __str__(self):      return self._name
  def __repr__(self):     return self._name
  # operator over-loading redirect to Expr
  def __add__(x,y):       return Expr(x).__add__(y,get_srcinfo(2))
  def __radd__(x,y):      return Expr(x).__radd__(y,get_srcinfo(2))
  def __mul__(x,y):       return Expr(x).__mul__(y,get_srcinfo(2))
  def __rmul__(x,y):      return Expr(x).__rmul__(y,get_srcinfo(2))
  def __getitem__(x,y):   return Expr(x).__getitem__(y,get_srcinfo(2))
  def proj(x,i):          return Expr(x).proj(get_srcinfo(2))

class IVar:
  """ Use this class to represent index variables """
  def __init__(self,nm):
    self._name    = nm
    self._srcinfo = get_srcinfo(2)
  def __str__(self):      return self._name
  def __repr__(self):     return self._name
  # operator over-loading redirect to IExpr
  def __add__(x,y):       return IExpr(x).__add__(y,get_srcinfo(2))
  def __radd__(x,y):      return IExpr(x).__radd__(y,get_srcinfo(2))
  def __mul__(x,y):       return IExpr(x).__mul__(y,get_srcinfo(2))
  def __rmul__(x,y):      return IExpr(x).__rmul__(y,get_srcinfo(2))
  def __lt__(x,y):        return IExpr(x).__lt__(y,get_srcinfo(2))
  def __gt__(x,y):        return IExpr(x).__gt__(y,get_srcinfo(2))
  def __le__(x,y):        return IExpr(x).__le__(y,get_srcinfo(2))
  def __ge__(x,y):        return IExpr(x).__ge__(y,get_srcinfo(2))
  def eq(x,y):            return IExpr(x).eq(y,get_srcinfo(2))

class Relation:
  """ Use this class to represent relation variables """
  def __init__(self,nm):
    self._name    = nm
    self._srcinfo = get_srcinfo(2)
  def __str__(self):      return self._name
  def __repr__(self):     return self._name
  def __call__(self,*args):
    srcinfo = get_srcinfo(2)
    idx = [ IExpr(a) for a in args ]
    return Pred( UST.Relation(self._name, idx, srcinfo) )

class Size:
  """ Must use this class to represent size variables """
  def __init__(self,nm):
    self._name    = Sym(nm)
    self._srcinfo = get_srcinfo(2)
  def __str__(self):      return str(self._name)
  def __repr__(self):     return repr(self._name)
  # operator over-loading redirect to IExpr
  def __add__(x,y):       return IExpr(x).__add__(y,get_srcinfo(2))
  def __radd__(x,y):      return IExpr(x).__radd__(y,get_srcinfo(2))
  def __mul__(x,y):       return IExpr(x).__mul__(y,get_srcinfo(2))
  def __rmul__(x,y):      return IExpr(x).__rmul__(y,get_srcinfo(2))
  def __lt__(x,y):        return IExpr(x).__lt__(y,get_srcinfo(2))
  def __gt__(x,y):        return IExpr(x).__gt__(y,get_srcinfo(2))
  def __le__(x,y):        return IExpr(x).__le__(y,get_srcinfo(2))
  def __ge__(x,y):        return IExpr(x).__ge__(y,get_srcinfo(2))
  def eq(x,y):            return IExpr(x).eq(y,get_srcinfo(2))


class Type:
  """ Type Constructions """
  def __init__(self,obj):
    typ = type(obj)
    if typ is Type:
      self._type = obj._type
    elif T.is_type(obj):
      self._type = obj
    elif obj is float:
      self._type = T.num
    elif typ is tuple:
      if len(obj) < 1:
        raise TypeError("Tuple types must have at least one entry")
      typs  = [ Type(t)._type for t in obj ]
      self._type = T.Tuple(None, typs)
    elif typ is list:
      if len(obj) < 1:
        raise TypeError("lists passed to Type must have at least one entry")
      # check that we have a list of pairs
      for i,pair in enumerate(obj):
        if type(pair) is not tuple or len(pair) != 2:
          raise TypeError(f"expected a list of (name,type) pairs, but "
                          f"entry {i} was not a pair")
      labels, typs = [],[]
      for i,(nm,typ) in enumerate(obj):
        if not is_valid_name(nm):
          raise TypeError(f"expected list of (name,type) pairs, but "
                          f"entry {i} did not have a valid name")
        labels.append(nm)
        typs.append(Type(typ)._type)
      self._type = T.Tuple(T.labels(labels),typs)
    else: raise TypeError("not recognized as a type: {typ}")

  def __getitem__(self,key):
    """ Tensor Type construction shorthand """
    if type(key) is not tuple: key = (key,)
    typ = self._type
    for k in reversed(key):
      if type(k) is Size:
        typ = T.Tensor(k._name,typ)
      elif is_pos_int(k):
        typ = T.Tensor(k,typ)
      else:
        raise TypeError("expected a range (Size or int))")
    return Type(typ)

  def namedtuple(self):
    if type(self._type) is not T.Tuple:
      raise TypeError("This is not a Tuple-type")
    return get_python_named_tuple(self._type)

class _BigOpClass:
  def __init__(self,op_ctr):
    self._op_ctr  = op_ctr
  def __getitem__(self,key):
    if not type(key) is tuple: key = (key,)
    binds   = []
    for k_slice in key:
      if type(k_slice) != slice:
        raise TypeError("Expected index arguments to have form 'i : n'")
      elif k_slice.step is not None:
        raise TypeError("Expected 'i : n', but not 'i : n : m'")

      if type(k_slice.start) is IVar:
        nm  = k_slice.start._name
      else:
        raise TypeError("Expected 'i' in 'i:n' to be an IVar")

      if type(k_slice.stop) is Size:
        rng = k_slice.stop._name
      elif is_pos_int(k_slice.stop):
        rng = k_slice.stop
      else:
        raise TypeError("expected 'n' in 'i:n' to be a range (Size or int))")

      # otherwise, we've got a proper abuse of the slicing feature
      # as if it was typing...
      binds.append( (nm,rng) )
    return _BigOpConstructor(self._op_ctr,binds)

class _BigOpConstructor:
  def __init__(self,op_ctr,binds):
    self._op_ctr  = op_ctr
    self._binds   = binds
  def __call__(self,body):
    op      = self._op_ctr
    srcinfo = get_srcinfo(2)
    # construct nested copies of the operator...
    body    = Expr(body)._ast
    for nm_rng in reversed(self._binds):
      body  = op(nm_rng[0], nm_rng[1], body, srcinfo)
    return Expr(body)

Sum = _BigOpClass(UST.Sum)
Gen = _BigOpClass(UST.Gen)

class _LetOpClass:
  def __getitem__(self,key):
    if not type(key) is tuple: key = (key,)
    if len(key)%2 != 0:
      raise TypeError("Expected an even number of arguments to 'Let'")
    srcinfo     = get_srcinfo(2)

    stmts       = []
    for i in range(0,len(key),2):
      # even argument...
      typ       = None
      if type(key[i]) is slice:
        if key[i].step is not None:
          raise TypeError(f"Expected argument {i} to be 'Var : Type', "
                          f"but not 'Var : Type : ??'")
        if type(key[i].start) is Var:
          name  = key[i].start._name
        else:
          raise TypeError(f"Expected 'x' in argument {i} 'x : Type' "
                          f"to be a Var")
        typ     = Type(key[i].stop)
      else:
        if type(key[i]) is Var:
          name  = key[i]._name
        else:
          raise TypeError(f"Expected argument {i} to be a Var")
      # odd argument...
      body      = Expr(key[i+1])

      stmts.append( UST.assign(name,typ,body,srcinfo) )

    return _LetBlock(stmts)

class _LetBlock:
  def __init__(self,stmts):
    self._stmts = stmts
  def __call__(self,expr):
    srcinfo = get_srcinfo(2)
    ret     = Expr(expr)
    return Expr(UST.Let( self._stmts, ret, srcinfo ))

Let = _LetOpClass()

def Tuple(*args,srcinfo=None):
  srcinfo = srcinfo or get_srcinfo(2)
  return Expr(UST.Tuple([ Expr(a)._ast for a in args ], srcinfo))

class Expr:
  """ Generic Expression """
  def __init__(self,obj):
    typ = type(obj)
    if   typ is Var:
      self._ast = UST.Var(obj._name, obj._srcinfo)
    elif typ is float or typ is int:
      self._ast = UST.Const(float(obj), null_srcinfo())
    elif typ is tuple:
      if len(obj) < 1:
        raise TypeError("tuple expressions must have at least one entry")
      args = [ a for a in obj ]
      self._ast = Tuple(*args,get_srcinfo(2))
    elif typ is Expr:
      self._ast = obj._ast
    elif isinstance(obj,UST.expr):
      self._ast = obj
    else: raise TypeError(f"unexpected argument type: {typ}")

  def __add__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Expr( UST.Add(lhs._ast, Expr(rhs)._ast, srcinfo) )
  def __radd__(rhs,lhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return __add__(Expr(lhs),rhs,srcinfo)

  def __mul__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    if isinstance(rhs,Pred):
      return rhs * lhs
    else:
      return Expr( UST.Mul(lhs._ast, Expr(rhs)._ast, srcinfo) )
  def __rmul__(rhs,lhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return __mul__(Expr(lhs),rhs,srcinfo)

  # Access
  def __getitem__(self,key,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    if not type(key) is tuple: key = (key,)
    idx = [ IExpr(k)._ast for k in key ]
    return Expr( UST.Access(self._ast, idx, srcinfo) )

  def proj(self, idx, srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    if type(idx) is not int and not is_valid_name(idx):
      raise TypeError("expected a number or label to project with")
    return Expr(UST.Proj(idx,self._ast,srcinfo))


class IExpr:
  """ Index Expressions """
  def __init__(self,obj):
    typ = type(obj)
    if   typ is IVar:
      self._ast = UST.IdxVar(obj._name, obj._srcinfo)
    elif typ is Size:
      self._ast = UST.IdxSize(obj._name, obj._srcinfo)
    elif typ is int:
      self._ast = UST.IdxConst(obj, null_srcinfo())
    elif typ is IExpr:
      self._ast = obj._ast
    elif isinstance(obj,UST.index):
      self._ast = obj
    else: raise TypeError(f"unexpected argument type: {typ}")

  def __add__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return IExpr( UST.IdxAdd(lhs._ast, IExpr(rhs)._ast, srcinfo) )
  def __radd__(rhs,lhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return __add__(Expr(lhs),rhs,srcinfo)

  def __mul__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    s = rhs
    if type(s) is int:
      s = Fraction(s,1)
    return IExpr( UST.IdxScale(s, lhs._ast, srcinfo) )
  def __rmul__(rhs,lhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return __mul__(rhs,lhs,srcinfo)

  def __lt__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Cmp('<',  lhs._ast, IExpr(rhs)._ast, srcinfo) )
  def __gt__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Cmp('>',  lhs._ast, IExpr(rhs)._ast, srcinfo) )
  def __le__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Cmp('<=', lhs._ast, IExpr(rhs)._ast, srcinfo) )
  def __ge__(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Cmp('>=', lhs._ast, IExpr(rhs)._ast, srcinfo) )
  def eq(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Cmp('==', lhs._ast, IExpr(rhs)._ast, srcinfo) )


class Pred:
  """ Predicate Expressions """
  def __init__(self,obj):
    if type(obj) is Pred:
      self._ast = obj._ast
    elif isinstance(obj,UST.pred):
      self._ast = obj
    else: raise TypeError("Should not directly construct a predicate...")

  def __mul__(p,e,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Expr( UST.Indicate(p._ast, Expr(e)._ast, srcinfo) )
  def __rmul__(p,e,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return __mul__(p,e,srcinfo)

  def conj(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Conj(lhs._ast,rhs._ast,srcinfo) )
  def disj(lhs,rhs,srcinfo=None):
    srcinfo = srcinfo or get_srcinfo(2)
    return Pred( UST.Disj(lhs._ast,rhs._ast,srcinfo) )

class _FunBuilderClass:
  def __getitem__(self,key):  return _NamedFun(None).__getitem__(key)
  def __call__(self,name):    return _NamedFun(name)

Fun = _FunBuilderClass()

class _NamedFun:
  def __init__(self,name):
    if name is not None and not is_valid_name(name):
      raise TypeError("expected valid name for function")
    self._name    = name
    self._srcinfo = get_srcinfo(3)

  def __getitem__(self,key):
    if not type(key) is tuple: key = (key,)

    vs, szs, rels = [], [], []
    arg_order     = []
    for k in key:
      if type(k) is Size:
        szs.append( UST.size_decl(k._name,k._srcinfo) )
        arg_order.append(str(k._name))

      elif type(k) is not slice or k.step is not None:
        raise TypeError(f"Expected all non-Size arguments to 'Fun' to "
                        f"have the form 'x : ?'")

      elif type(k.start) is Var:
        nm  = k.start._name
        typ = Type(k.stop)._type
        vs.append( UST.var_decl(nm,typ,k.start._srcinfo) )
        arg_order.append(nm)

      elif type(k.start) is Relation:
        nm    = k.start._name
        if type(k.stop) is not tuple:
          raise TypeError("Expected 'T' after 'Relation : T' to be a tuple")
        rngs  = []
        for r in k.stop:
          if type(r) is Size:
            rngs.append(r._name)
          elif is_pos_int(k):
            rngs.append(k)
          else:
            raise TypeError("expected a range (Size or int))")
        rels.append( UST.rel_decl(nm,rngs,k.start._srcinfo) )
        arg_order.append(nm)

      else:
        raise TypeError(f"argument name of unrecognized type {type(k.start)}")

    return _SignedFun(self._name, arg_order, vs, szs, rels, self._srcinfo)

class _SignedFun:
  def __init__(self, name, arg_order, vs, szs, rels, srcinfo):
    self._name      = name
    self._arg_order = arg_order
    self._vars      = vs
    self._sizes     = szs
    self._relations = rels
    self._srcinfo   = srcinfo

  def __call__(self,body):
    f = UST.function(name       = self._name,
                     arg_order  = self._arg_order,
                     vars       = self._vars,
                     sizes      = self._sizes,
                     relations  = self._relations,
                     body       = Expr(body)._ast,
                     srcinfo    = self._srcinfo )
    return Function(f)



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# some lightweight tests during development...

if __name__ == '__main__':
  def dot_ust():
    x   = 'x'
    y   = 'y'
    i   = 'i'
    n   = Sym('n')
    NS  = null_srcinfo()

    body = UST.Sum(i,n,
              UST.Mul( UST.Access(UST.Var(x,NS),
                                  [UST.IdxVar(i,NS)], NS),
                       UST.Access(UST.Var(y,NS),
                                  [UST.IdxVar(i,NS)], NS), NS ), NS)

    f = UST.function( name  = 'dot',
                      arg_order = ['n','x','y'],
                      vars  = [UST.var_decl(x, T.Tensor(n,T.num), NS),
                               UST.var_decl(y, T.Tensor(n,T.num), NS)],
                      sizes = [UST.size_decl(n, NS)],
                      relations = [],
                      body  = body,
                      srcinfo   = NS )

    ast = f.typecheck()
    print(str(ast))
  dot_ust()

  def dot_shorthand():
    x, y  = Var('x'), Var('y')
    i     = IVar('i')
    n     = Size('n')
    num   = Type(float)

    f = Fun('dot')[ n, x : num[n], y : num[n] ](
      Sum[i:n]( x[i] * ((i < n) * y[i+1]) )
    )
    print(f)

  dot_shorthand()


