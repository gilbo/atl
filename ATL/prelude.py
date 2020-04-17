
from re import compile as _re_compile
from inspect import (currentframe as _curr_frame,
                     getframeinfo as _get_frame_info)

def is_pos_int(obj):
  return type(obj) is int and obj >= 1

_valid_pattern = _re_compile("^[a-zA-Z_]\w*$")
def is_valid_name(obj):
  return (type(obj) is str) and (_valid_pattern.match(obj) != None)

class Sym:
  _unq_count   = 1

  def __init__(self,nm):
    if not is_valid_name(nm):
      raise TypeError(f"expected an alphanumeric name string, "
                      f"but got '{nm}'")
    self._nm    = nm
    self._id    = Sym._unq_count
    Sym._unq_count += 1

  def __str__(self):
    return self._nm

  def __repr__(self):
    return f"{self._nm}${self._id}"

  def __hash__(self): return id(self)

  def __lt__(lhs,rhs): return (lhs._nm,lhs._id) < (rhs._nm,rhs._id)

  def name(self):
    return self._nm

  def copy(self):
    return Sym(self._nm)

class Context:
  def __init__(self):
    self._envs   = [{}]
  def push(self):
    self._envs.insert(0,{})
  def pop(self):
    self._envs.pop(0)
  def set(self, nm, val):
    self._envs[0][nm] = val
  def get(self, nm):
    for e in self._envs:
      if nm in e:
        return e[nm]
    return None
  def items(self):
    for e in self._envs:
      for nm,val in e.items():
        yield nm,val

# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)

class SrcInfo:
  def __init__(self,filename,lineno,col_offset=None,
                    end_lineno=None,end_col_offset=None,
                    function=None):
    self.filename       = filename
    self.lineno         = lineno
    self.col_offset     = col_offset
    self.end_lineno     = end_lineno
    self.end_col_offset = end_col_offset
    self.function       = function
  def __str__(self):
    colstr = "" if self.col_offset is None else f":{self.col_offset}"
    return f"{self.filename}:{self.lineno}{colstr}"

def get_srcinfo(depth=1):
  f = _curr_frame()
  for k in range(0,depth): f = f.f_back
  finfo = _get_frame_info(f)
  filename, lineno, function = finfo.filename, finfo.lineno, finfo.function
  del f, finfo
  return SrcInfo(filename, lineno, function)

_null_srcinfo_obj = SrcInfo("unknown",0)
def null_srcinfo(): return _null_srcinfo_obj





# Contexts
class Environment:
  """Replacement for Dict with ability to keep a stack"""
  def __init__(self, init_dict=None):
    self._bottom_dict   = init_dict
    self._stack         = [dict()]

  def push(self):
    self._stack.append(dict())

  def pop(self):
    self._stack.pop()

  def __getitem__(self,key):
    for e in reversed(self._stack):
      if key in e:  return e[key]
    if self._bottom_dict and key in self._bottom_dict:
      return self._bottom_dict[key]
    raise KeyError(key)

  def __contains__(self,key):
    for e in reversed(self._stack):
      if key in e: return True
    return bool(self._bottom_dict and key in self._bottom_dict)

  def __setitem__(self,key,val):
    self._stack[-1][key] = val



# map [str] -> [str]
def wrapjoin(strs, col_width=79, delimit="" ):
  rlines      = []
  count       = 0
  line        = []
  def flushline():
    nonlocal  line, count
    rlines.append( delimit.join(line) )
    count     = 0
    line      = []

  for s in strs:
    count    += len(s)
    # if we overran the line, first thing to do is to flush it
    if len(line) > 0 and count > col_width:
      flushline()
      count  += len(s)
    else:
      count  += len(delimit)
    # add this string to the current line regardless
    line.append(s)

  if len(line) > 0:
    flushline()
  return rlines


