
from prelude import *

import atl_types as T
from frontend import UST, AST
from functions import Function

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Quoted Code Objects for constructing via external meta-programming

class Var:
  """ Use this class to represent variables """
  def __init__(self,nm):  self._name = nm
  def __str__(self):      return self._name
  def __repr__(self):     return self._name
  # operator over-loading redirect to Expr
  def __add__(x,y):       return Expr(x) + y
  def __radd__(x,y):      return y + Expr(x)
  def __mul__(x,y):       return Expr(x) * y
  def __rmul__(x,y):      return y * Expr(x)
  def __getitem__(x,y):   return Expr(x).__getitem__(y)
  def fst(x):             return Expr(x).fst()
  def snd(x):             return Expr(x).snd()

class IVar:
  """ Use this class to represent index variables """
  def __init__(self,nm):  self._name = nm
  def __str__(self):      return self._name
  def __repr__(self):     return self._name
  # operator over-loading redirect to IExpr
  def __add__(x,y):       return IExpr(x) + y
  def __radd__(x,y):      return y + IExpr(x)
  def __mul__(x,y):       return IExpr(x) * y
  def __rmul__(x,y):      return y * IExpr(x)
  def __lt__(x,y):        return IExpr(x) < y
  def __gt__(x,y):        return IExpr(x) > y
  def __le__(x,y):        return IExpr(x) <= y
  def __ge__(x,y):        return IExpr(x) >= y
  def __eq__(x,y):        return IExpr(x) == y

class Relation:
  """ Use this class to represent relation variables """
  def __init__(self,nm):  self._name = nm
  def __str__(self):      return self._name
  def __repr__(self):     return self._name
  def __call__(self,*args):
    idx = [ IExpr(a) for a in args ]
    return Pred( UST.Relation(self._name, idx) )

class Size:
  """ Must use this class to represent size variables """
  def __init__(self,nm):  self._name = Sym(nm)
  def __str__(self):      return str(self._name)
  def __repr__(self):     return repr(self._name)
  # operator over-loading redirect to IExpr
  def __add__(x,y):       return IExpr(x) + y
  def __radd__(x,y):      return y + IExpr(x)
  def __mul__(x,y):       return IExpr(x) * y
  def __rmul__(x,y):      return y * IExpr(x)
  def __lt__(x,y):        return IExpr(x) < y
  def __gt__(x,y):        return IExpr(x) > y
  def __le__(x,y):        return IExpr(x) <= y
  def __ge__(x,y):        return IExpr(x) >= y
  def __eq__(x,y):        return IExpr(x) == y


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
      if len(obj) != 2:
        raise TypeError("Tuple types must be pairs")
      self._type = T.Pair( Type(obj[0])._type, Type(obj[1])._type )
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
    op = self._op_ctr
    # construct nested copies of the operator...
    body = Expr(body)._ast
    for nm_rng in reversed(self._binds):
      body = op(nm_rng[0], nm_rng[1], body)
    return Expr(body)

Sum = _BigOpClass(UST.Sum)
Gen = _BigOpClass(UST.Gen)

class _LetOpClass:
  def __getitem__(self,key):
    if not type(key) is tuple: key = (key,)
    if len(key)%2 != 0:
      raise TypeError("Expected an even number of arguments to 'Let'")

    stmts = []
    for i in range(0,len(key),2):
      # even argument...
      typ = None
      if type(key[i]) is slice:
        if key[i].step is not None:
          raise TypeError(f"Expected argument {i} to be 'Var : Type', "
                          f"but not 'Var : Type : ??'")
        if type(key[i].start) is Var:
          name  = key[i].start._name
        else:
          raise TypeError(f"Expected 'x' in argument {i} 'x : Type' "
                          f"to be a Var")
        typ   = Type(key[i].stop)
      else:
        if type(key[i]) is Var:
          name  = key[i]._name
        else:
          raise TypeError(f"Expected argument {i} to be a Var")
      # odd argument...
      body  = Expr(key[i+1])

      stmts.append( UST.assign(name,typ,body) )

    return _LetBlock(stmts)

class _LetBlock:
  def __init__(self,stmts):
    self._stmts = stmts
  def __call__(self,expr):
    ret   = Expr(expr)
    return Expr(UST.Let( self._stmts, ret ))

Let = _LetOpClass()

def Pair(fst,snd):
  return Expr(UST.Pair(Expr(fst)._ast,Expr(snd)._ast))

class Expr:
  """ Generic Expression """
  def __init__(self,obj):
    typ = type(obj)
    if   typ is Var:
      self._ast = UST.Var(obj._name)
    elif typ is float or typ is int:
      self._ast = UST.Const(float(obj))
    elif typ is tuple:
      if len(obj) != 2:
        raise TypeError("tuple expressions must be pairs")
      self._ast = Pair(obj[0],obj[1])
    elif typ is Expr:
      self._ast = obj._ast
    elif isinstance(obj,UST.expr):
      self._ast = obj
    else: raise TypeError(f"unexpected argument type: {typ}")

  def __add__(lhs,rhs):
    return Expr( UST.Add(lhs._ast, Expr(rhs)._ast) )
  def __radd__(rhs,lhs): return __add__(Expr(lhs),rhs)

  def __mul__(lhs,rhs):
    if isinstance(rhs,Pred):
      return rhs * lhs
    else:
      return Expr( UST.Mul(lhs._ast, Expr(rhs)._ast) )
  def __rmul__(rhs,lhs): return __mul__(Expr(lhs),rhs)

  # Access
  def __getitem__(self,key):
    if not type(key) is tuple: key = (key,)
    idx = [ IExpr(k)._ast for k in key ]
    return Expr( UST.Access(self._ast, idx) )

  def fst(self):
    return Expr(UST.Proj(0,self._ast))

  def snd(self):
    return Expr(UST.Proj(1,self._ast))


class IExpr:
  """ Index Expressions """
  def __init__(self,obj):
    typ = type(obj)
    if   typ is IVar:
      self._ast = UST.IdxVar(obj._name)
    elif typ is Size:
      self._ast = UST.IdxSize(obj._name)
    elif typ is int:
      self._ast = UST.IdxConst(obj)
    elif typ is IExpr:
      self._ast = obj._ast
    elif isinstance(obj,UST.index):
      self._ast = obj
    else: raise TypeError(f"unexpected argument type: {typ}")

  def __add__(lhs,rhs):
    return IExpr( UST.IdxAdd(lhs._ast, IExpr(rhs)._ast) )
  def __radd__(rhs,lhs): return __add__(Expr(lhs),rhs)

  def __mul__(lhs,rhs):
    s = rhs
    if type(s) is int:
      s = Fraction(s,1)
    return IExpr( UST.IdxScale(s, lhs._ast) )
  def __rmul__(rhs,lhs): return __mul__(rhs,lhs)

  def __lt__(lhs,rhs):
    return Pred( UST.Cmp('<', lhs._ast, IExpr(rhs)._ast) )
  def __gt__(lhs,rhs):
    return Pred( UST.Cmp('>', lhs._ast, IExpr(rhs)._ast) )
  def __le__(lhs,rhs):
    return Pred( UST.Cmp('<=', lhs._ast, IExpr(rhs)._ast) )
  def __ge__(lhs,rhs):
    return Pred( UST.Cmp('>=', lhs._ast, IExpr(rhs)._ast) )
  def __eq__(lhs,rhs):
    return Pred( UST.Cmp('==', lhs._ast, IExpr(rhs)._ast) )


class Pred:
  """ Predicate Expressions """
  def __init__(self,obj):
    if type(obj) is Pred:
      self._ast = obj._ast
    elif isinstance(obj,UST.pred):
      self._ast = obj
    else: raise TypeError("Should not directly construct a predicate...")

  def __mul__(p,e):
    return Expr( UST.Indicate(p._ast, Expr(e)._ast) )
  def __rmul__(p,e): return __mul__(p,e)

  def conj(lhs,rhs):
    return Pred( UST.Conj(lhs._ast,rhs._ast) )
  def disj(lhs,rhs):
    return Pred( UST.Disj(lhs._ast,rhs._ast) )

class _FunBuilderClass:
  def __getitem__(self,key):  return _NamedFun(None).__getitem__(key)
  def __call__(self,name):    return _NamedFun(name)

Fun = _FunBuilderClass()

class _NamedFun:
  def __init__(self,name):
    if name is not None and not is_valid_name(name):
      raise TypeError("expected valid name for function")
    self._name = name

  def __getitem__(self,key):
    if not type(key) is tuple: key = (key,)

    vs, szs, rels = [], [], []
    for k in key:
      if type(k) is Size:
        szs.append(k._name)

      elif type(k) is not slice or k.step is not None:
        raise TypeError(f"Expected all non-Size arguments to 'Fun' to "
                        f"have the form 'x : ?'")

      elif type(k.start) is Var:
        nm  = k.start._name
        typ = Type(k.stop)._type
        vs.append( UST.var_decl(nm,typ) )

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
        rels.append( UST.rel_decl(nm,rngs) )

      else:
        raise TypeError(f"argument name of unrecognized type {type(k.start)}")

    return _SignedFun(self._name, vs, szs, rels)

class _SignedFun:
  def __init__(self, name, vs, szs, rels):
    self._name      = name
    self._vars      = vs
    self._sizes     = szs
    self._relations = rels

  def __call__(self,body):
    f = UST.function(name       = self._name,
                     vars       = self._vars,
                     sizes      = self._sizes,
                     relations  = self._relations,
                     body       = Expr(body)._ast )
    return Function(f)



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# some lightweight tests during development...

if __name__ == '__main__':
  def dot_ust():
    x = 'x'
    y = 'y'
    i = 'i'
    n = Sym('n')

    body = UST.Sum(i,n, UST.Mul( UST.Access( UST.Var(x), [UST.IdxVar(i)] ),
                                 UST.Access( UST.Var(y), [UST.IdxVar(i)] ) ))

    f = UST.function( name  = 'dot',
                      vars  = [UST.var_decl(x, T.Tensor(n,T.num)),
                              UST.var_decl(y, T.Tensor(n,T.num))],
                      sizes = [n],
                      relations = [],
                      body  = body)

    ast = f.typecheck()
    print(str(ast))
  dot_ust()

  def dot_shorthand():
    x, y  = Var('x'), Var('y')
    i     = IVar('i')
    n     = Size('n')
    num   = Type(float)

    f = Fun('dot')[ n, x : num[n], y : num[n] ](
      Sum[i:n]( x[i] * y[i] )
    )
    print(f)

  dot_shorthand()


