
from .prelude import *

from . import atl_types as T

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
    labels  = typ.labels
    if labels is None:
      labels = [ f"_{i}" for i,t in enumerate(typ.types) ]
    _nm_tpl_cache = namedtuple('ATL_Tuple',labels)
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
    labels = typ.labels or [ f"_{i}" for i,t in enumerate(typ.types) ]
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
      raise TypeError(f"{pre} expected numpy.ndarray of 64-bit floats")
    shape   = typ.shape()
    # substitute size values in...
    shape   = [ rng if type(rng) is int else ctxt.get(rng)
                for rng in shape ]
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
      for k in range(i+1,Ndim): t_stride *= shape[k]
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
  if val.dtype != bool:
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
    for k in range(i+1,Ndim): t_stride *= shape[k]
    np_stride = val.strides[i]
    if t_stride != np_stride:
      raise TypeError(f"{pre} expected tight ndarray layout with "
                      f"dimensions nested right/outermost "
                      f"to left/innermost")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Type to output buffer allocation

def get_numpy_buffers(ctxt, typ):
  pre = "Bad Output Type: "
  if not typ.is_SoA():
    raise TypeError(f"{pre} tuples inside arrays are currently unsupported")

  elif typ is T.num:
    # allocate a 1-entry array
    return np.zeros([1])

  elif type(typ) is T.Tuple:
    args  = [ get_numpy_buffers(ctxt, t) for t in typ.types ]
    ctr   = get_python_named_tuple(typ)
    return ctr(args)

  elif type(typ) is T.Tensor:
    shape = typ.shape()
    shape = [ rng if type(rng) is int else ctxt.get(rng)
              for rng in shape ]
    return np.zeros(shape)

  else: assert False, "invalid type case"
