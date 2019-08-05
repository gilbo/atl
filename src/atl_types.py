
import sys
from adt import ADT
from adt import memo as ADTmemo

from prelude import Sym

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

_Types = ADT("""
module Types {
  type  = Num     ()
        | Error   ()
        | Pair    ( type fst, type snd )
        | Tensor  ( range range, type type )
} """, {
    'range': lambda x: (type(x) is int and x >= 1) or (type(x) is Sym),
})
ADTmemo(_Types,['Num','Error','Pair','Tensor'],{'range':id})

Num     = _Types.Num
Error   = _Types.Error
Pair    = _Types.Pair
Tensor  = _Types.Tensor
num     = Num()
error   = Error()

# --------------------------------------------------------------------------- #
# string representation of types...

def _Type_String(t):
  if   type(t) is Num:
    return "Num"
  elif type(t) is Error:
    return "Error"
  elif type(t) is Pair:
    return f"({t.fst},{t.snd})"
  elif type(t) is Tensor:
    rs = []
    while type(t) is Tensor:
      rs.append(t.range)
      t = t.type
    rngs = ",".join([ str(r) for r in rs ])
    return f"[{rngs}]{t}"
  else: assert false, "impossible type case"
_Types.type.__str__ = _Type_String
del _Type_String

# --------------------------------------------------------------------------- #
# Type introspection

def is_type(obj):
  return isinstance(obj,_Types.type)

def _type_has_pairs(t):
  tclass = type(t)
  if   tclass is Pair:    return True
  elif tclass is Tensor:  return t.type.has_pairs()
  else:                   return False
_Types.type.has_pairs = _type_has_pairs
del _type_has_pairs

# --------------------------------------------------------------------------- #
# Struct-of-Arrays transform on types

def _typ_SoA_transform(t,rngs=[]):
  tclass = type(t)
  if t is num or t is error:
    # possibly unroll the ranges stack here
    for r in reversed(rngs):
      t = Tensor(r,t)
    return t
  elif tclass is Tensor:
    rngs = rngs.copy()
    rngs.append(t.range)
    return t.type.SoA_transform(rngs)
  elif tclass is Pair:
    return Pair(  t.fst.SoA_transform(rngs),
                  t.snd.SoA_transform(rngs)  )
  else: assert false, "impossible type case"
_Types.type.SoA_transform = _typ_SoA_transform
del _typ_SoA_transform



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Tests

if __name__ == '__main__':
  assert is_type(num)
  assert is_type(error)
  assert num == Num()
  assert error == Error()

  nn    = Pair(num,num)
  vec3  = Tensor(3,num)
  assert is_type(nn)
  assert is_type(vec3)
  assert nn == Pair(num,num)
  assert vec3 == Tensor(3,num)

  nn3   = Tensor(3,nn)
  v3v3  = Pair(vec3,vec3)
  assert nn.has_pairs()
  assert not vec3.has_pairs()
  assert nn3.has_pairs()
  assert v3v3.has_pairs()

  assert v3v3 == nn3.SoA_transform()



