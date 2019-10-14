
import sys
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

_Types = ADT("""
module Types {
  type  = Num     ()
        | Error   ()
        | Tuple   ( labels? labels, type* types )
        | Tensor  ( range   range,  type  type  )

  labels = ( name* names )

} """, {
    'range':  lambda x: is_pos_int(x) or type(x) is Sym,
    'name':   is_valid_name,
})
ADTmemo(_Types,['Num','Error','Tuple','Tensor','labels'],{
  'range':  lambda x: x,
  'name':   lambda x: x,
})

Num     = _Types.Num
Error   = _Types.Error
Tuple   = _Types.Tuple
labels  = _Types.labels
Tensor  = _Types.Tensor
num     = Num()
error   = Error()

# extra constructor...
def tensor_shape(ranges):
  typ = num
  for r in reversed(ranges):
    typ = Tensor(r,typ)
  return typ

# --------------------------------------------------------------------------- #
# string representation of types...

@extclass(_Types.type)
def __str__(t):
  if not hasattr(t,'_str_cached'):
    if   type(t) is Num:
      t._str_cached = "Num"
    elif type(t) is Error:
      t._str_cached = "Error"
    elif type(t) is Tuple:
      if t.labels is None:
        entries = ','.join([ str(st) for st in t.types ])
      else:
        entries = ','.join([ f"{nm}={st}"
                             for nm,st in zip(t.labels.names,t.types) ])
      t._str_cached = f"({entries})"
    elif type(t) is Tensor:
      rs    = []
      base  = t
      while type(base) is Tensor:
        rs.append(base.range)
        base = base.type
      rngs = ",".join([ str(r) for r in rs ])
      t._str_cached = f"[{rngs}]{base}"
    else: assert False, "impossible type case"
  return t._str_cached
del __str__

# --------------------------------------------------------------------------- #
# Type introspection

def is_type(obj):
  return isinstance(obj,_Types.type)

@extclass(_Types.type)
def has_tuples(t):
  tclass = type(t)
  if   tclass is Tuple:   return True
  elif tclass is Tensor:  return t.type.has_tuples()
  else:                   return False
del has_tuples

@extclass(_Types.type)
def is_SoA(t):
  tclass = type(t)
  if tclass is Tuple:
    for subt in t.types:
      if not subt.is_SoA():
        return False
    return True
  elif tclass is Tensor:
    return not t.has_tuples()
  else:
    return True
del is_SoA

@extclass(Tensor)
def shape(t):
  shp = []
  while type(t) is Tensor:
    shp.append(t.range)
    t = t.type
  return shp
del shape

# --------------------------------------------------------------------------- #
# Type Matching to account for name differences between Tuples
# We take a hybrid nominal/structural approach by allowing
# two types to match so long as their names are not _conflicting_

@extclass(_Types.type)
def matches(ltyp, rtyp):
  lclass, rclass = type(ltyp), type(rtyp)
  if lclass is not rclass:
    return False
  elif lclass is Num or lclass is Error:
    return True
  elif lclass is Tensor:
    return ltyp.type.matches(rtyp.type)
  elif lclass is Tuple:
    if ltyp.labels is not None and rtyp.labels is not None:
      if len(ltyp.labels.names) != len(rtyp.labels.names):
        return False
      for lnm,rnm in zip(ltyp.labels.names, rtyp.labels.names):
        if lnm != rnm:
          return False
    # whether names were absent or all consistent, we must still
    # check consistency of types
    if len(ltyp.types) != len(rtyp.types):
      return False
    for lt,rt in zip(ltyp.types, rtyp.types):
      if not lt.matches(rt):
        return False
    # passed all checks
    return True
  else: assert False, "impossible type case"
del matches


# --------------------------------------------------------------------------- #
# Struct-of-Arrays transform on types

@extclass(_Types.type)
def SoA_transform(t,rngs=[]):
  tclass  = type(t)
  if t is num or t is error:
    # unroll the ranges stack here
    for r in reversed(rngs):
      t   = Tensor(r,t)
    return t
  elif tclass is Tensor:
    rngs  = rngs.copy()
    rngs.append(t.range)
    return t.type.SoA_transform(rngs)
  elif tclass is Tuple:
    subs  = [ st.SoA_transform(rngs) for st in t.types ]
    return Tuple( t.labels, subs )
  else: assert False, "impossible type case"
del SoA_transform

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
