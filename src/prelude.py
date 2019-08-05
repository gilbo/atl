
from re import compile as _re_compile

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

# from a github gist by victorlei
def extclass(cls):
  return lambda f: (setattr(cls,f.__name__,f) or f)