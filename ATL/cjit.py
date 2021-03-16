
from .prelude import *
from . import atl_types as T

import ctypes
import os
import sys
import time
import subprocess

import hashlib

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Helpers and directory setup

def _shell(cstr):
  subprocess.run(cstr, check=True, shell=True)

_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_C_CACHE        = os.path.join(_HERE_DIR,'.atl_c_cache')
if not os.path.isdir(_C_CACHE):
  os.mkdir(_C_CACHE)

def get_time(s):
  if not os.path.exists(s):
    return None
  else:
    return os.path.getmtime(s)

# clean the cache when it exceeds 50MB;
# keep all files created in the last day or so
def clean_cache(size_trigger = int(50e6), clear_time_window = 86400.0):
  curr_time   = time.time()
  filenames   = [ os.path.join(_C_CACHE,f) for f in os.listdir(_C_CACHE) ]
  used_size   = sum(os.path.getsize(f) for f in filenames)

  if used_size > size_trigger:
    for file in filenames:
      mtime   = get_time(file)
      if curr_time - mtime > clear_time_window:
        os.remove(file)

# try out a single clean every time we load this module
clean_cache()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


_typcache = {}
def get_typstr(typ):
  if typ == T.num:
    return "double", None
  elif type(typ) is T.Tensor:
    assert not typ.has_tuples()
    return "double *", None
  elif type(typ) is T.Tuple:
    if typ in _typcache:
      return _typcache[typ]
    # otherwise build the struct type
    typname     = re.sub('[^0-9a-zA-Z]+', '_', str(typ))

    subtyps     = [ get_typstr(t)[0] for t in typ.types ]
    fields      = [ f"_{i}" for i in range(0,len(subtyps)) ]
    entries     = [ f"  {t} {f};" for t,f in zip(subtyps, fields) ]

    if typ.labels:
      altfields   = typ.labels.names
      altentries  = [ f"  {t} {f};" for t, f in zip(subtyps, altfields) ]
      new_struct  = (f"struct {typname} {{"+"\n"+
                      "  union {\n"+
                      "   struct {\n"+
                      "    "+'\n    '.join(entries)+"\n"+
                      "   };\n"+
                      "   struct {\n"+
                      "    "+'\n    '.join(altentries)+"\n"+
                      "   };\n"+
                      "  };\n"+
                      "};\n")
    else:
      new_struct  = (f"struct {typname} {{"+"\n"+
                      '\n'.join(entries)+"\n"+
                      "};\n")

    #self._typdefs.add(new_struct)
    _typcache[typ] = (typname, new_struct)
    return (typname, new_struct)
  else: assert False, "unexpected case"


_ctype_struct_cache = {}
def get_ctype(typ):
  """ translate double,bool,int or an ATL type into a C-type """
  if typ == float or typ == T.num:
    return ctypes.c_double
  elif typ == int:
    return ctypes.c_int
  elif typ == bool:
    return ctypes.c_bool
  elif type(typ) is T.Tensor:
    assert typ.is_SoA()
    return ctypes.POINTER(ctypes.c_double)
  elif type(typ) is T.Tuple:
    if typ in _ctype_struct_cache:
      return _ctype_struct_cache[typ]

    name, defstr  = get_typstr(typ)
    fields        = [ (f"_{i}", get_ctype(t))
                      for i,t in enumerate(typ.types) ]
    # create a new type
    struct        = type(name, (ctypes.Structure,), { '_fields_': fields })

    _ctype_struct_cache[typ] = struct
    return struct
  else: assert False, "bad case"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class CJitSig:
  def __init__(self, size_names, var_typs, n_rels, ret_typ):
    self.size_names   = size_names
    self.var_typs     = var_typs
    self.n_rels       = n_rels
    self.ret_typ      = ret_typ


class CJit:
  """ Manage JIT compilation of C code derived from ATL
  """

  def __init__(self, name, type_sig, cpp_str, cpp_flags=""):
    assert type(name) is str
    assert type(type_sig) is CJitSig
    assert type(cpp_str) is str
    assert type(cpp_flags) is str

    self._name      = name
    self._type_sig  = type_sig
    self._cpp_str   = cpp_str

    hashstr       = hashlib.md5(cpp_str.encode('utf-8')).hexdigest()

    fname         = name + hashstr
    cpp_filename  = os.path.join(_C_CACHE,f"{fname}.cpp")
    so_filename   = os.path.join(_C_CACHE,f"{fname}.so")
    comp_cmd      = (f"clang++ -Wall -Werror -fPIC -O3 -shared -std=c++11 "
                     f"{cpp_flags}"
                     f"-o {so_filename} {cpp_filename}")

    def matches_file(src, fname):
      if not os.path.isfile(fname):
        return False
      else:
        with open(fname, 'r', encoding = 'utf-8') as F:
          return F.read() == src
    def write_file(src, fname):
      with open(fname, 'w', encoding = 'utf-8') as F:
        F.write(src)
    def is_so_fresh():
      so_time   = get_time(so_filename)
      cpp_time  = get_time(cpp_filename)
      if so_time is None or cpp_time is None:
        return False
      else:
        return so_time >= cpp_time

    # do we need to rebuild the corresponding SO?
    rebuild = not is_so_fresh()
    if not matches_file(cpp_str, cpp_filename):
      write_file(cpp_str, cpp_filename)
      rebuild = True
    if rebuild:
      #print(comp_cmd)
      _shell(comp_cmd)

    # load the module regardless
    module        = ctypes.CDLL(so_filename)
    self._module  = module
    self._cfun    = getattr(module, name)

    #wrap the module function with types
    atyps = []
    for _ in type_sig.size_names:
      atyps.append(get_ctype(int))
    for a in type_sig.var_decls:
      atyps.append(get_ctype(a.type))
    atyps.append(ctypes.POINTER(get_ctype(type_sig.ret_typ)))
    for _ in range(0,type_sig.n_rels):
      atyps.append(ctypes.POINTER(ctypes.c_bool))
    self._cfun.argtypes   = atyps
    self._cfun.restype    = None

  def __call__(self, vs, sizes, relations, out):

    def pack_input(val,typ,is_output=False):
      if typ == T.num:
        if is_output:
          assert type(val) is np.ndarray and val.shape == (1,)
          return ctypes.c_double(val[0])
        else:
          assert type(val) is float
          return ctypes.c_double(val)
      elif type(typ) is T.Tensor:
        assert type(val) is np.ndarray
        ptr       = val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return ptr
      elif type(typ) is T.Tuple:
        assert isinstance(val, tuple) and len(val) == len(typ.types)
        struct    = get_ctype(typ)
        return struct(*[ pack_input(v,t,is_output)
                         for v,t in zip(val,typ.types) ])
      else: assert False, "bad case"

    # collect all the arguments to the function call
    fargs         = []
    for sz in sizes:
      fargs.append(ctypes.c_int(sz))
    for v,vd in zip(vs, self._func.vars):
      fargs.append(pack_input(v,vd.type))
    # special case the packing of the output argument...
    out_obj = pack_input(out, self._func.rettype, is_output=True)
    fargs.append(ctypes.byref(out_obj))
    # back to the relation data now...
    for r in relations:
      assert type(r) is np.ndarray
      fargs.append( r.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)) )

    # do the C-function invocation through the FFI
    self._cfun(*fargs)

    def pack_output(obj,argval,typ):
      if typ == T.num:
        assert type(argval) is np.ndarray and argval.shape == (1,)
        #print(obj)
        #print(type(obj))
        if type(obj) is float:
          argval[0] = obj
        else:
          argval[0] = obj.value
      elif type(typ) is T.Tensor:
        assert type(argval) is np.ndarray
        pass # no action necessary, data written via pointer already
      elif type(typ) is T.Tuple:
        for i,(a,t) in enumerate(zip(argval,typ.types)):
          pack_output( getattr(obj,f"_{i}"), a, t )
      else: assert False, "bad case"

    pack_output(out_obj, out, self._type_sig.ret_typ)

