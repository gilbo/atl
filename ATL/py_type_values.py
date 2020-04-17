
from .prelude import *

from . import atl_types as T
from . import frontend  as F

import numpy as np
from collections import namedtuple

from weakref import WeakKeyDictionary

"""
py_type_values.py

This module describes how ATL types are associated to Python values/types
so that ATL functions may be called from Python.  This includes functions
to check whether a Python value conforms to a given ATL type, construct
default values/allocations from a type, etc.

- Tensors are mapped to numpy arrays
- Tuples  are mapped to collections.namedtuple

"""

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Named Tuple Cache

_nm_tpl_cache = WeakKeyDictionary({})

def get_python_named_tuple(typ):
  if type(typ) is not T.Tuple:
    raise TypeError("expected a Tuple-type")
  if typ not in _nm_tpl_cache:
    if typ.labels is None:
      labels = [ f"_{i}" for i,t in enumerate(typ.types) ]
    else:
      labels  = typ.labels.names
    _nm_tpl_cache[typ] = namedtuple('ATL_Tuple',labels,rename=True)
  return _nm_tpl_cache[typ]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Argument value checking at Python boundary

def argcheck_python_value(ctxt, typ, val, arg_id, is_output=False):
  pre = f"Bad Argument - {arg_id}:"
  if not typ.is_SoA():
    raise TypeError(f"{pre} tuples inside arrays are currently unsupported")

  elif typ is T.num:
    # output scalars should be buffers
    if is_output:
      if type(val) is not np.ndarray:
        raise TypeError(f"{pre} expected numpy.ndarray")
      elif val.dtype != float and val.dtype != np.float64:
        raise TypeError(f"{pre} expected numpy.ndarray of 64-bit floats")
      if val.ndim != 1 or val.shape[0] != 1:
        raise TypeError(f"{pre} expected shape [1], got {val.shape}")
        
    # input scalars should be raw floats
    elif type(val) is not float:
      raise TypeError(f"{pre} expected float")

  elif type(typ) is T.Tuple:
    labels = ( typ.labels.names if typ.labels is not None else
                [ f"_{i}" for i,t in enumerate(typ.types) ] )
    if not isinstance(val, tuple):
      raise TypeError(f"{pre} expected tuple")
    elif len(val) != len(typ.types):
      raise TypeError(f"{pre} expected tuple size {len(typ.types)}, "
                      f"got {len(val)}")
    else:
      for l,t,v in zip(labels,typ.types,val):
        argcheck_python_value(ctxt, t, v, f"{arg_id}.{l}",
                              is_output=is_output)

  elif type(typ) is T.Tensor:
    if type(val) is not np.ndarray:
      raise TypeError(f"{pre} expected numpy.ndarray")
    elif val.dtype != float and val.dtype != np.float64:
      raise TypeError(f"{pre} expected numpy.ndarray of 64-bit floats, "
                      f"but got numpy.ndarray of {val.dtype}")
    shape   = typ.shape()
    # substitute size values in...
    shape   = [ rng if type(rng) is int else ctxt.get(rng)
                for rng in shape ]
    npshape = val.shape
    Ndim    = len(shape)
    if Ndim != val.ndim:
      raise TypeError(f"{pre} expected {Ndim} dimensions, got {val.ndim}")
    # check correspondence
    if tuple(shape) != tuple(npshape):
      raise TypeError(f"{pre} expected shape {shape}, got {list(npshape)}")
    # stride packing check
    for i,(td,npd) in enumerate(zip(shape,npshape)):
      t_stride  = 1
      for k in range(0,i): t_stride *= shape[k]
      assert val.strides[i] % val.itemsize == 0, 'sanity'
      np_stride = val.strides[i] // val.itemsize
      if t_stride != np_stride:
        raise TypeError(f"{pre} expected tight ndarray layout with "
                        f"dimensions nested right/outermost "
                        f"to left/innermost")

  else: assert False, "invalid type case"

def argcheck_python_size(ctxt, val, arg_id):
  pre = f"Bad Argument - {arg_id}:"
  if type(val) is not int:
    raise TypeError(f"{pre} expected int")

def argcheck_python_relation(ctxt, sizes, val, arg_id):
  pre = f"Bad Argument - {arg_id}:"
  if type(val) is not np.ndarray:
    raise TypeError(f"{pre} expected numpy.ndarray")
  elif val.dtype != bool:
    raise TypeError(f"{pre} expected numpy.ndarray of 8-bit bools")
  assert val.itemsize == 1, 'bools should be 1 byte'

  # substitute size values in...
  shape   = [ rng if type(rng) is int else ctxt.get(rng)
              for rng in sizes ]
  npshape = val.shape
  Ndim    = len(shape)
  if Ndim != val.ndim:
    raise TypeError(f"{pre} expected {Ndim} dimensions, got {val.ndim}")
  # check correspondence
  for i,(td,npd) in enumerate(zip(shape,npshape)):
    if td != npd:
      raise TypeError(f"{pre} expected shape {shape}, got {list(npshape)}")
  # stride packing check
  for i,(td,npd) in enumerate(zip(shape,npshape)):
    t_stride  = 1
    for k in range(0,i): t_stride *= shape[k]
    np_stride = val.strides[i]
    if t_stride != np_stride:
      raise TypeError(f"{pre} expected tight ndarray layout with "
                      f"dimensions nested right/outermost "
                      f"to left/innermost")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Argument value conversion to code constants

def pyval_to_ast(ctxt, typ, val, srcinfo):
  assert typ.is_const_size(), "TODO: implement non-const-size conversion"

  if typ == T.num:
    assert type(val) == float
    return F.AST.Const( val, typ, srcinfo )
  elif type(typ) is T.Tuple:
    assert len(val) == len(typ.types)
    args  = [ pyval_to_ast(ctxt, t, v, srcinfo)
              for t,v in zip(list(val), typ.types) ]
    return F.AST.Tuple( args, typ, srcinfo )
  elif type(typ) is T.Tensor:
    assert type(val) is np.ndarray
    assert val.dtype == float or val.dtype == np.float64
    assert val.shape[0] == typ.range
    args  = [ pyval_to_ast(ctxt, typ.type, v, srcinfo) for v in val ]
    return F.AST.TensorLit( args, typ.type, srcinfo )
  else: error("unexpected impossible case")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Type to output buffer allocation

def get_numpy_buffers(ctxt, typ):
  pre = "Bad Output Type: "
  if not typ.is_SoA():
    raise TypeError(f"{pre} tuples inside arrays are currently unsupported")

  elif typ is T.num:
    # allocate a 1-entry array
    return np.zeros([1], order='F')

  elif type(typ) is T.Tuple:
    args  = [ get_numpy_buffers(ctxt, t) for t in typ.types ]
    ctr   = get_python_named_tuple(typ)
    return ctr(*args)

  elif type(typ) is T.Tensor:
    shape = typ.shape()
    shape = [ rng if type(rng) is int else ctxt.get(rng)
              for rng in shape ]
    return np.zeros(shape, order='F')

  else: assert False, "invalid type case"

