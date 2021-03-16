
from .adt import ADT

from .prelude import *
from . import builtins as B
from . import atl_types as T

from collections import ChainMap, namedtuple, OrderedDict
import itertools
import math
from fractions import Fraction
import re

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# helpers

nonedict = lambda: defaultdict(lambda: None)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# C_Lang description

_binops = {
  "+":    "+",
  "-":    "-",
  "*":    "*",
  "/":    "/",
  #
  "and":  "&&",
  "or":   "||",
  #
  "!=":   "!=",
  "==":   "==",
  "<":    "<",
  ">":    ">",
  "<=":   "<=",
  ">=":   ">=",
}
_prec = {
  #
  "or":   10,
  "and":  20,
  #
  "!=":   30,
  "==":   30,
  #
  "<":    40,
  ">":    40,
  "<=":   40,
  ">=":   40,
  #
  "+":    50,
  "-":    50,
  "*":    60,
  "/":    60,
}

C = ADT("""
module C_Lang {
    proc    = ( name       name
                sym*       sizes,    -- int32 values
                fnarg*     args,
                relarg*    relargs,
                stmt*      body,
                srcinfo    srcinfo )

    fnarg   = ( sym name, type type, srcinfo srcinfo )
    relarg  = ( sym name, shape shape, srcinfo srcinfo )

    stmt    = Assign    ( access lhs, expr rhs )
            | Reduce    ( access lhs, expr rhs )
            | If        ( expr cond, stmt* body )
            | For       ( sym iter, expr lo, expr hi, stmt* body )
            | Alloc     ( sym name, type type )
            attributes( srcinfo srcinfo )

    expr   = Read       ( access acc )
           | Const      ( object val )
           | BinOp      ( binop op, expr lhs, expr rhs )
           | BuiltIn    ( builtin f, expr* args )
           attributes( object type, srcinfo srcinfo )

    access = ( sym name, expr* idx, int* proj )
}
""", {
    'name':     is_valid_name,
    'op':       lambda x: x in _binops,
    'type':     T.is_type,
    'builtin':  lambda x: isinstance(x, B.BuiltIn),
    'binop':    lambda x: x in _binops,
    'sym':      lambda x: type(x) is Sym,
    'type':     T.is_type,
    'shape':    lambda x: (type(x) is tuple and
                           all(is_pos_int(r) or type(r) is Sym for r in x)),
    'srcinfo':  lambda x: type(x) is SrcInfo,
})

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

_typcache = {}
def get_type(typ):
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

    subtyps     = [ get_type(t)[0] for t in typ.types ]
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



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def h_include():
  return ''

def c_include():
  return ('#include <cstdlib>\n'+
          '#include <cmath>\n'+
          '#include <iostream>\n'
          #'#define MAX(a,b) std::fmax(a,b)\n'+
          #'#define MIN(a,b) std::fmin(a,b)\n'+
          'using namespace std;\n')

def compile_procs(h_file, c_file, procs):
  structs = OrderedDict()
  fdecls  = []
  fdefs   = []
  for p in procs:
    CS = CodeString(p)
    for x in CS.getstructs():
      structs[x] = True
    fdefs.append(CS.getfunc())
    fdecls.append(CS.getdecl())
  structs = [ x for x in structs ]

  H_str = [f'{h_include()}',
           '\n',
           '\n'.join(structs),
           '\n',
           '\n'.join(fdecls),
           '\n']
  C_str = [f'extern "C" {{\n',
           f'  #include "{h_file}"\n',
           f'}};\n',
           f'{c_include()}',
           '\n',
           '\n'.join(fdefs),
           '\n']

  return ('\n'.join(H_str), '\n'.join(C_str))
  #with open(h_file, 'w') as Hfile:
  #  for s in H_str:
  #    Hfile.write(s)
  #with open(c_file, 'w') as Cfile:
  #  for s in C_str:
  #    Cfile.write(s)

class CodeString:
  def __init__(self, proc):
    assert type(proc) is C.proc
    self._proc        = proc
    self._ctxt        = ChainMap({})
    self._names       = ChainMap({})
    self._shapes      = {}

    # determine free variable sets so that we
    # can free buffers intelligently
    #     self._fv_cache[stmt] is the set of FV __AFTER__ stmt
    self._fv_cache    = {}
    self.build_free_vars(proc.body)
    # tracked while compiling
    self._live_bufs   = set()

    #self._typcache    = {}
    self._typdefs     = OrderedDict()
    self._tab         = ""
    self._lines       = []

    argdefs, argnms   = self.gen_args(proc)
    self.push_scope()
    for s in proc.body:
      self.gen_stmt(s)
    self.pop_scope()

    fndef   = (f"void {proc.name}({','.join(argdefs)}) {{"+"\n"+
                #'  cout << "START" << endl;\n'+
                '\n'.join(self._lines)+"\n"+
                #'  cout << "STOP" << endl;\n'+
                "}")

    fndecl  = (f"void {proc.name}({','.join(argdefs)});")

    self._func  = fndef
    self._fdecl = fndecl

  def getfunc(self):
    return self._func

  def getdecl(self):
    return self._fdecl

  def getstructs(self):
    return [ x for x in self._typdefs ]

  def line(self, s):
    self._lines.append(f"{self._tab}{s}")

  def push_scope(self, name_only=False, tab_only=False):
    if not name_only:
      self._tab  += "  "
    if not tab_only:
      self._ctxt  = self._ctxt.new_child()
      self._names = self._names.new_child()

  def pop_scope(self):
    self._tab   = self._tab[0:-2] # trim two chars off the end
    self._ctxt  = self._ctxt.parents
    self._names = self._names.parents

  def build_free_vars(self, stmts):
    """ run this once at the top level to build the
        full cache of free buffer variables for lookups"""
    def fvexpr(e, FVs):
      if type(e) is C.Read:
        FVs.add(e.acc.name)
      elif type(e) is C.Const:
        pass
      elif type(e) is C.BinOp:
        fvexpr(e.lhs, FVs)
        fvexpr(e.rhs, FVs)
      elif type(e) is C.BuiltIn:
        for a in e.args:
          fvexpr(a, FVs)
      elif type(e) is C.Select:
        fvexpr(e.cond,    FVs)
        fvexpr(e.iftrue,  FVs)
        fvexpr(e.iffalse, FVs)
      else:
        assert False, "bad case"
    def fvstmts(stmts, FVs):
      for s in reversed(stmts):
        self._fv_cache[s] = FVs
        FVs = FVs.copy()
        if type(s) is C.Assign or type(s) is C.Reduce:
          fvexpr(s.rhs, FVs)
          FVs.add(s.lhs.name)
        elif type(s) is C.If or type(s) is C.For:
          # must run twice to ensure convergence of FV set
          # internally to the loop
          FVs = fvstmts(s.body, FVs)
          FVs = fvstmts(s.body, FVs)
        elif type(s) is C.Alloc:
          if s.name in FVs:
            FVs.remove(s.name)
        else: assert False, "bad case"
      return FVs

    fvstmts(stmts, set())

  def gen_shape(self, typ):
    assert typ.is_SoA()
    if type(typ) is T.Tuple:
      return tuple( self.gen_shape(t) for t in typ.types )
    else:
      return typ.shape_or_scalar()

  def gen_args(self, proc):
    defs, nms = [], []
    for sz in proc.sizes:
      nm  = self.new_name(sz)
      nms.append(nm)
      defs.append(f"int {nm}")
    for a in proc.args:
      nm  = self.new_name(a.name)
      self._shapes[a.name] = self.gen_shape(a.type)
      nms.append(nm)
      typ = self.get_type(a.type)
      if nm == "output":
        defs.append(f"{typ}* {nm}")
      else:
        defs.append(f"{typ} {nm}")
    for r in proc.relargs:
      nm  = self.new_name(r.name)
      self._shapes[r.name] = r.shape
      nms.append(nm)
      defs.append(f"bool* {nm}")

    return (defs, nms)

  def gen_stmt(self, stmt):
    styp = type(stmt)

    # common case of scalar assignment to a fresh name
    #if styp is C.Assign and not self.is_namedef(stmt.lhs.name):
    #  assert (stmt.rhs.type == float and len(stmt.lhs.idx) == 0
    #                                 and len(stmt.lhs.proj) == 0)
    #  name  = self.new_name(stmt.lhs.name)
    #  rhs   = self.gen_expr(stmt.rhs)
    #  self.line(f"double /*{repr(stmt.lhs.name)}*/ {name} = {rhs};")

    if styp is C.Assign or styp is C.Reduce:
      op  = "=" if styp is C.Assign else "+="
      lhs = self.acc_expr(stmt.lhs)
      rhs = self.gen_expr(stmt.rhs)
      self.line(f"{lhs} {op} {rhs};")

    elif styp is C.If:
      cond  = self.gen_expr(stmt.cond)
      self.line(f"if ({cond}) {{")
      self.push_scope()
      for bs in stmt.body:
        self.gen_stmt(bs)
      self.pop_scope()
      self.line('}')

    elif styp is C.For:
      lo    = self.gen_expr(stmt.lo)
      hi    = self.gen_expr(stmt.hi)
      self.push_scope(name_only=True)
      itr   = self.new_name(stmt.iter)
      self.line(f"for(int {itr}={lo}; {itr} < {hi}; {itr}++) {{")
      self.push_scope(tab_only=True)
      for bs in stmt.body:
        self.gen_stmt(bs)
      self.pop_scope()
      self.line('}')

    elif styp is C.Alloc:
      name  = self.new_name(stmt.name)
      self._shapes[stmt.name] = stmt.type.shape_or_scalar()
      if stmt.type == T.num:
        self.line(f"double {name} = 0;")
      else:
        self._live_bufs.add(stmt.name)
        szexpr = self.buf_size(stmt.type)
        self.line(f"double *{name} = (double*)(calloc({szexpr},8));")

    else: assert False, "bad case"

    # attempt to free any live buffers that are
    # no longer needed after this statement
    FV = self._fv_cache[stmt]
    #self.line(f"// free vars: {' '.join([ str(x) for x in FV ])}")
    to_free = self._live_bufs - FV
    #self.line(f"// to_free vars: {' '.join([ str(x) for x in to_free ])}")
    for nm in to_free:
      #self.line(f'cout << "free {nm}" << endl;')
      self.line(f"free({self._ctxt[nm]});")
    self._live_bufs -= to_free

  def gen_expr(self, e, prec=0):
    etyp = type(e)

    if etyp is C.Read:
      return self.acc_expr(e.acc)
    elif etyp is C.Const:
      if e.type == bool:
        return "false" if e.val == False else "true"
      elif e.type == int:
        return str(e.val)
      elif e.type == float:
        return str(e.val)
      else: assert False, "bad Const type case"
    elif etyp is C.BinOp:
      op  = _binops[e.op]
      pop = _prec[e.op]
      lhs = self.gen_expr(e.lhs, prec=pop)
      rhs = self.gen_expr(e.rhs, prec=pop+1)
      res = f"{lhs} {op} {rhs}"
      if pop < prec:
        res = f"({res})"
      return res
    elif etyp is C.BuiltIn:
      args  = [ self.gen_expr(a) for a in e.args ]
      return e.f.C_compile(*args)
    else: assert False, "bad case"

  def shape_strs(self, shp):
    # case 1: we have some structs on the outside of this shape
    if len(shp) > 0 and (type(shp[0]) is tuple or type(shp[0]) is list):
      return tuple( self.shape_strs(e) for e in shp )
    # case 2: we are through the structs and now characterizing a tensor
    else:
      return tuple( str(r) if is_pos_int(r) else self._ctxt[r]
                    for r in shp )

  def buf_size(self, typ):
    shp = self.shape_strs(typ.shape())
    return '*'.join(shp)

  def acc_expr(self, acc):
    #if acc.name not in self._ctxt:
    #  print('UH OH', acc.name, acc)
    #  base = f"/* {repr(acc.name)} */" + str(acc.name)
    #else:
    base      = self._ctxt[acc.name]
    if base == "output":
      base    = "(*output)"
    if len(acc.idx) > 0 or len(acc.proj) > 0:
      shp     = self.shape_strs(self._shapes[acc.name])
    # reduce out projections if present
    for i in acc.proj:
      base    = f"{base}._{i}"
      shp     = shp[i]

    if len(acc.idx) == 0:
      return base
    else:
      idx     = [ self.gen_expr(i,prec=_prec['*']) for i in acc.idx ]
      iters   = list(zip(reversed(idx),reversed(shp)))
      res     = iters[0][0]
      for i,r in iters[1:]:
        res = f"({res})"
        res   = f"{i} + {r}*{res}"
      return f"{base}[{res}]"

  def get_type(self, typ):
    typname, typdef   = get_type(typ)

    # make sure to cache the used tuple structs...
    if type(typ) is T.Tuple:
      for t in typ.types:
        self.get_type(t)
      self._typdefs[typdef] = True

    return typname

  def is_namedef(self, nm):
    return (nm in self._ctxt)

  def new_name(self, nm):
    assert nm not in self._ctxt
    nmstr = str(nm)
    if nmstr not in self._names:
      # the first occurrence of nmstr is undecorated
      retstr = nmstr
    else:
      # if we've already allocated nmstr, then
      # we must allocate a fresh name via a count decoration
      retstr  = self._names[nmstr]
      # iterate the following process till we are sure we have a fresh name
      while retstr in self._names:
        m     = re.match('^(.*)_([0-9]*)$', retstr)
        if not m:
          # retstr was undecorated
          retstr = retstr + "_1"
        else:
          # if retstr was already decorated, increment the counter
          retstr = f"{m[1]}_{int(m[2]) + 1}"

    # update to the latest renaming of this name string
    # and also mark that the renamed string is being used
    assert retstr not in self._names
    self._names[nmstr]  = retstr
    self._names[retstr] = retstr
    self._ctxt[nm]      = retstr
    return retstr
