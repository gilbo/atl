
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *
import numpy as np
import ctypes

from .halide import halide_type_t, halide_buffer_t, halide_dimension_t
from .halide import hw_expr_t, hw_var_t, hw_rdom_t
from .halide import hw_func_t, hw_img_t, hw_param_t
from .halide import C

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# HIR description

_binops = {
    "+":    True,
    "-":    True,
    "*":    True,
    "/":    True,
    "and":  True,
    "or":   True,
    "!=":   True,
    "==":   True,
    "<":    True,
    ">":    True,
    "<=":   True,
    ">=":   True,
}

_basetypes = {
    "f32":  True,
    "f64":  True,
}
for bw in ["8","16","32","64"]:
    _basetypes["i"+bw] = True
    _basetypes["u"+bw] = True

HIR = ADT("""
module HIR {
    pipeline   = Pipeline  ( var*   vars,
                             rdom*  rdoms,
                             func*  funcs,
                             param* params,
                             img*   imgs,
                             stmt*  stmts,
                             func*  outputs )

    var    = Var       ( string name )
    rdom   = RDom      ( string name, range* bounds )
    param  = Param     ( string name, type typ )
    img    = Img       ( string name, int ndim, type typ )
    func   = Func      ( string name, int ndim, type typ )
           | ImgFunc   ( img img )

    expr   = Const     ( object v, type typ )
           | Evar      ( var v )
           | Erdom     ( rdom r )
           | Eparam    ( param p )
           | BinOp     ( op op, expr lhs, expr rhs )
           | MathFn1   ( string name, expr arg )
           | Min       ( expr lhs, expr rhs )
           | Max       ( expr lhs, expr rhs )
           | Clamp     ( expr val, expr lo, expr hi )
           | Pow       ( expr base, expr exp )
           | ATan2     ( expr y, expr x )
           | Select    ( expr pred, expr lhs, expr rhs )
           | FAccess   ( func f, expr* args )
           | BigSum    ( rdom r, expr body )

    stmt   = PureDef   ( func f, var* args, expr rhs )
           | Update    ( func f, expr* args, expr rhs )

    range  = Range ( expr lo, expr extent )
    type   = Type  ( typbase base, int lanes )
}
""", {
    'op':         lambda x:  x in _binops,
    'typbase':    lambda x:  x in _basetypes,
})
ADTmemo(HIR,['Type'],{'typbase': lambda x: x})
f32, f64            =  HIR.Type('f32',1), HIR.Type('f64',1)
u8,  u16, u32, u64  = (HIR.Type('u8',1),  HIR.Type('u16',1),
                       HIR.Type('u32',1), HIR.Type('u64',1))
i8,  i16, i32, i64  = (HIR.Type('i8',1),  HIR.Type('i16',1),
                       HIR.Type('i32',1), HIR.Type('i64',1))
for nm in _basetypes:
  setattr(HIR,nm,HIR.Type(nm,1))

# Module Export control
__all__ = [ 'HIR' ]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# HIR Readable String Format

@extclass(HIR.Type)
def __str__(t):
    if t.lanes > 1: return f"{t.base}v{t.lanes}"
    else:           return t.base
@extclass(HIR.Var)
def __str__(v): return v.name
@extclass(HIR.RDom)
def __str__(r):
    bds = ']['.join([ f"{rg.lo},{rg.extent}" for rg in r.bounds ])
    return f"{r.name}[{bds}]"
@extclass(HIR.Param)
def __str__(p): return f"{p.name} : {p.typ}"
@extclass(HIR.Img)
def __str__(i): return f"{i.name}({i.ndim}) : {i.typ}"
@extclass(HIR.func)
def __str__(f):
    if   type(f) is HIR.Func:    return f.name
    elif type(f) is HIR.ImgFunc: return f.img.name

_HIR_op_prec = {
    "+"   : 30,
    "-"   : 30,
    "*"   : 40,
    "/"   : 40,
    "<"   : 20,
    ">"   : 20,
    "<="  : 20,
    ">="  : 20,
    "=="  : 10,
    "!="  : 10,
    "and" : 5,
    "or"  : 5,
}
@extclass(HIR.expr)
def _str_rep(e,prec=0):
    eclass = type(e)
    s      = "ERROR"
    if   eclass is HIR.Evar:   s = e.v.name
    elif eclass is HIR.Erdom:  s = e.r.name
    elif eclass is HIR.Eparam: s = e.p.name
    elif eclass is HIR.Const:
        if e.typ is i32 or e.typ is f64:
            s = str(e.v)
        else:
            s = f"({e.v}:{e.typ})"
    elif eclass is HIR.BinOp:
        op_prec = _HIR_op_prec[e.op]
        s = (f"{e.lhs._str_rep(op_prec)} {e.op} {e.rhs._str_rep(op_prec+1)}")
        if prec > op_prec: s = f"({s})"
    elif eclass is HIR.MathFn1:
        s = f"{e.name}({e.arg})"
    elif eclass is HIR.Min or eclass is HIR.Max:
        fname = "Min" if eclass is HIR.Min else "Max"
        s = f"{fname}({e.lhs},{e.rhs})"
    elif eclass is HIR.Clamp:
        s = f"Clamp({e.val},{e.lo},{e.hi})"
    elif eclass is HIR.Pow:
        s = f"pow({e.base},{e.exp})"
    elif eclass is HIR.ATan2:
        s = f"atan2({e.y},{e.x})"
    elif eclass is HIR.Select:
        s = f"select({e.pred},{e.lhs},{e.rhs})"
    elif eclass is HIR.FAccess:
        args = ','.join([ str(a) for a in e.args ])
        s = f"{e.f}({args})"
    elif eclass is HIR.BigSum:
        s = f"sum({e.r.name},{e.body})"
    return s
@extclass(HIR.expr)
def __str__(e): return e._str_rep()

@extclass(HIR.stmt)
def __str__(s):
    args = ','.join([ str(a) for a in s.args ])
    return f"{s.f}({args}) = {s.rhs}"

@extclass(HIR.Pipeline)
def __str__(p):
    vs    = '\n        '.join([ str(v) for v in p.vars ])
    rs    = '\n        '.join([ str(r) for r in p.rdoms ])
    fs    = '\n        '.join([ str(f) for f in p.funcs ])
    ps    = '\n        '.join([ str(p) for p in p.params ])
    imgs  = '\n        '.join([ str(i) for i in p.imgs ])
    stmts = '\n        '.join([ str(s) for s in p.stmts ])
    outs  = '\n        '.join([ str(o) for o in p.outputs ])

    s = (f"Var     {vs}\n"
         f"RDom    {rs}\n"
         f"Func    {fs}\n"
         f"Param   {ps}\n"
         f"Img     {imgs}\n"
         f"Stmt    {stmts}\n"
         f"Out     {outs}")
    return s

@extclass(HIR.Pipeline)
def signature_str(self):
    ps    = '\n        '.join([ str(p) for p in self.params ])
    imgs  = '\n        '.join([ str(i) for i in self.imgs ])
    outs  = '\n        '.join([ str(o) for o in self.outputs ])

    s = (f"Param   {ps}\n"
         f"Img     {imgs}\n"
         f"Out     {outs}")
    return s
del signature_str

del __str__, _str_rep

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# NumPy to Halide bindings

def _ndarray_to_halide_buf(a):
    def typ_convert(dt):
        t = dt.type
        # remapping to prevent some pointless errors
        if t is float:
            t = np.float64 if (sys.float_info.max_exp == 1024) else np.float32
        if t is int:
            t = np.int32
        if t is bool or t is np.bool_:
            t = np.uint8
        # main case switch
        if   t is np.int8:    return halide_type_t(C.type_int,8,1)
        elif t is np.int16:   return halide_type_t(C.type_int,16,1)
        elif t is np.int32:   return halide_type_t(C.type_int,32,1)
        elif t is np.int64:   return halide_type_t(C.type_int,64,1)
        elif t is np.uint8:   return halide_type_t(C.type_uint,8,1)
        elif t is np.uint16:  return halide_type_t(C.type_uint,16,1)
        elif t is np.uint32:  return halide_type_t(C.type_uint,32,1)
        elif t is np.uint64:  return halide_type_t(C.type_uint,64,1)
        elif t is np.float32: return halide_type_t(C.type_float,32,1)
        elif t is np.float64: return halide_type_t(C.type_float,64,1)
        else:
            raise TypeError(f"unexpected type {t}")

    buf  = halide_buffer_t()
    buf.device              = 0
    buf.device_interface    = None
    buf.host                = a.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    buf.flags               = 0
    buf.type                = typ_convert(a.dtype)
    buf.dimensions          = a.ndim
    buf.dim                 = (halide_dimension_t * a.ndim)()
    # now loop through and sort out each dimension
    for k in range(0,a.ndim):
        s = int(a.strides[k] / a.itemsize)
        assert a.strides[k] % a.itemsize == 0
        buf.dim[k] = halide_dimension_t(0,a.shape[k],s,0)
    buf.padding             = None

    return buf

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# HIR Pipeline Compilation

@extclass(HIR.Type)
def struct(typ):
    if hasattr(typ, '_cached_struct'): return typ._cached_struct
    if   typ.base[0] == 'f': flag = C.type_float
    elif typ.base[0] == 'u': flag = C.type_uint
    elif typ.base[0] == 'i': flag = C.type_int
    bits = int(typ.base[1:])
    typ._cached_struct = halide_type_t(flag,bits,typ.lanes)
    return typ._cached_struct
del struct

@extclass(HIR.Type)
def ctype(typ):
    if hasattr(typ, '_cached_ctype'): return typ._cached_ctype
    if   typ.base == 'f32': ct = ctypes.c_float
    elif typ.base == 'f64': ct = ctypes.c_double
    elif typ.base == 'u8':  ct = ctypes.c_uint8
    elif typ.base == 'u16': ct = ctypes.c_uint16
    elif typ.base == 'u32': ct = ctypes.c_uint32
    elif typ.base == 'u64': ct = ctypes.c_uint64
    elif typ.base == 'i8':  ct = ctypes.c_int8
    elif typ.base == 'i16': ct = ctypes.c_int16
    elif typ.base == 'i32': ct = ctypes.c_int32
    elif typ.base == 'i64': ct = ctypes.c_int64
    typ._cached_ctype = ct
    return typ._cached_ctype
del ctype

class _HIR_Compilation:
    def __init__(self, pipe):
        self._pipe   = pipe
        self._vars   = {
            v : C.hwrap_new_var(bytes(v.name,'utf-8'))
            for v in pipe.vars }
        self._params = {
            p : C.hwrap_new_param(bytes(p.name,'utf-8'),
                                  p.typ.struct())
            for p in pipe.params }
        self._imgs   = {
            i : C.hwrap_new_img(bytes(i.name,'utf-8'),
                                i.ndim, i.typ.struct())
            for i in pipe.imgs }
        self._funcs  = {
            f : C.hwrap_new_func(bytes(f.name,'utf-8'))
            for f in pipe.funcs }
        self._rdoms  = {}
        self._exprs  = {}
        self._stmts  = {
            s : self.get_stmt(s)
            for s in pipe.stmts }

    def get_func(self,f):
        if f in self._funcs: return self._funcs[f]
        else:
            assert type(f) is HIR.ImgFunc
            f_img   = C.hwrap_img_to_func(self._imgs[f.img])
            self._funcs[f] = f_img
            return f_img

    def get_rdom(self,r):
        if r in self._rdoms: return self._rdoms[r]
        n_bd    = len(r.bounds)
        bds     = []
        for rng in r.bounds:
            bds.append( self.get_expr(rng.lo) )
            bds.append( self.get_expr(rng.extent) )
        c_bds   = ((n_bd * 2) * hw_expr_t)(*bds)
        rd = C.hwrap_new_rdom(bytes(r.name,'utf-8'),
                              n_bd, c_bds)
        self._rdoms[r] = rd
        return rd

    def get_expr(self,e):
        if e in self._exprs: return self._exprs[e]
        ee      = None
        eclass  = type(e)
        if   eclass is HIR.Const:
            assert e.typ.lanes == 1
            ee  = getattr(C, f"hwrap_{e.typ.base}_to_expr")(e.v)
        elif eclass is HIR.Evar:
            ee  = C.hwrap_var_to_expr(self._vars[e.v])
        elif eclass is HIR.Erdom:
            rr  = self.get_rdom(e.r)
            ee  = C.hwrap_rdom_to_expr(rr)
        elif eclass is HIR.Eparam:
            ee  = C.hwrap_param_to_expr(self._params[e.p])
        elif eclass is HIR.BinOp:
            if   e.op == "+":   op_f = C.hwrap_add
            elif e.op == "-":   op_f = C.hwrap_sub
            elif e.op == "*":   op_f = C.hwrap_mul
            elif e.op == "/":   op_f = C.hwrap_div
            elif e.op == "and": op_f = C.hwrap_and
            elif e.op == "or":  op_f = C.hwrap_or
            elif e.op == "==":  op_f = C.hwrap_eq
            elif e.op == "!=":  op_f = C.hwrap_neq
            elif e.op == "<":   op_f = C.hwrap_lt
            elif e.op == ">":   op_f = C.hwrap_gt
            elif e.op == "<=":  op_f = C.hwrap_le
            elif e.op == ">=":  op_f = C.hwrap_ge
            else: assert False, f"unrecognized operator: {e.op}"
            ee  = op_f(self.get_expr(e.lhs), self.get_expr(e.rhs))
        elif eclass is HIR.MathFn1:
            if   e.name == "sin":   fn = C.hwrap_sin
            elif e.name == "cos":   fn = C.hwrap_cos
            elif e.name == "tan":   fn = C.hwrap_tan
            elif e.name == "asin":  fn = C.hwrap_asin
            elif e.name == "acos":  fn = C.hwrap_acos
            elif e.name == "atan":  fn = C.hwrap_atan
            elif e.name == "log":   fn = C.hwrap_log
            elif e.name == "exp":   fn = C.hwrap_exp
            elif e.name == "sqrt":  fn = C.hwrap_sqrt
            else: assert False, f"unrecognized math function: {e.name}"
            ee  = fn(self.get_expr(e.arg))
        elif eclass is HIR.Min:
            ee  = C.hwrap_min(self.get_expr(e.lhs), self.get_expr(e.rhs))
        elif eclass is HIR.Max:
            ee  = C.hwrap_max(self.get_expr(e.lhs), self.get_expr(e.rhs))
        elif eclass is HIR.Clamp:
            ee  = C.hwrap_clamp(self.get_expr(e.val),
                              self.get_expr(e.lo), self.get_expr(e.hi))
        elif eclass is HIR.Pow:
            ee  = C.hwrap_pow(self.get_expr(e.base), self.get_expr(e.exp))
        elif eclass is HIR.ATan2:
            ee  = C.hwrap_atan2(self.get_expr(e.y), self.get_expr(e.x))
        elif eclass is HIR.Select:
            ee  = C.hwrap_select(self.get_expr(e.pred),
                                 self.get_expr(e.lhs),
                                 self.get_expr(e.rhs))
        elif eclass is HIR.FAccess:
            args    = [ self.get_expr(a) for a in e.args ]
            c_args  = (len(args)*hw_expr_t)(*args)
            ee  = C.hwrap_access_func(self.get_func(e.f),
                                      len(args),c_args)
        elif eclass is HIR.BigSum:
            r   = self.get_rdom(e.r)
            ee  = C.hwrap_big_sum(r, self.get_expr(e.body))
        self._exprs[e]  = ee
        return ee

    def get_stmt(self, s):
        f       = self.get_func(s.f)
        rhs     = self.get_expr(s.rhs)
        if type(s) is HIR.PureDef:
            args    = [ self._vars[v] for v in s.args ]
            c_args  = (len(args)*hw_var_t)(*args)
            return C.hwrap_pure_def(f, len(args), c_args, rhs)
        elif type(s) is HIR.Update:
            args    = [ self.get_expr(e) for e in s.args ]
            c_args  = (len(args)*hw_expr_t)(*args)
            return C.hwrap_update(f, len(args), c_args, rhs)


class _HIR_JIT_Execution(_HIR_Compilation):
    def __init__(self, pipe):
        super().__init__(pipe)

    def _check_args(self, params, imgs, outputs):
        if type(params) != list: raise TypeError("expected list of 'params'")
        if type(imgs)   != list: raise TypeError("expected list of 'imgs'")
        if type(outputs)!= list: raise TypeError("expected list of 'outputs'")
        n_p = len(self._pipe.params)
        n_i = len(self._pipe.imgs)
        n_o = len(self._pipe.outputs)
        if len(params)  != n_p: raise TypeError(f"expected list of {n_p} 'params'")
        if len(imgs)    != n_i: raise TypeError(f"expected list of {n_i} 'imgs'")
        if len(outputs) != n_o: raise TypeError(f"expected list of {n_o} 'outputs'")

    def _set_params(self, params):
        for k,val in enumerate(params):
            p_IR    = self._pipe.params[k]
            p_obj   = self._params[p_IR]
            val_ref = ctypes.byref(p_IR.typ.ctype()(val))
            C.hwrap_set_param(p_obj,val_ref)
            assert p_IR.typ.lanes == 1
            val_expr = getattr(C,f"hwrap_{p_IR.typ.base}_to_expr")(val)
            C.hwrap_set_param_estimate(p_obj,val_expr)

    def _jit_compile(self, params, imgs, outputs):
        if hasattr(self,'_pipeline_obj'): return
        i32 = C.hwrap_i32_to_expr

        self._set_params(params)

        # estimate input bounds
        for k,np_arr in enumerate(imgs):
            img_IR  = self._pipe.imgs[k]
            img_obj = self._imgs[img_IR]
            for d_i,d_N in enumerate(np_arr.shape):
                C.hwrap_set_img_bound_estimate(img_obj,d_i,i32(0),i32(d_N))

        # estimate output bounds
        out_fs  = []
        for k,np_arr in enumerate(outputs):
            out_IR  = self._pipe.outputs[k]
            out_obj = self._funcs[out_IR]
            out_fs.append(out_obj)
            for d_i,d_N in enumerate(np_arr.shape):
                C.hwrap_set_func_bound_estimate(out_obj,d_i,i32(0),i32(d_N))

        n_out   = len(outputs)
        c_fs    = (n_out * hw_func_t)(*out_fs)
        self._pipeline_obj = C.hwrap_new_pipeline(n_out,c_fs)
        C.hwrap_autoschedule_pipeline(self._pipeline_obj)
        #C.hwrap_pipeline_print_loop_nest(self._pipeline_obj)

    def _exec(self, params, imgs, outputs):
        #print('executing...')
        #print(self._pipe)
        self._check_args(params,imgs,outputs)
        self._jit_compile(params,imgs,outputs)

        # bind the arguments
        self._set_params(params)

        # bind input buffers
        for k,np_arr in enumerate(imgs):
            h_buf   = _ndarray_to_halide_buf(np_arr)
            img_obj = self._imgs[self._pipe.imgs[k]]
            C.hwrap_set_img(img_obj, ctypes.byref(h_buf))

        # bind output buffers and Execute!
        outs    = [ _ndarray_to_halide_buf(arr) for arr in outputs ]
        c_outs  = (len(outs)*halide_buffer_t)(*outs)
        C.hwrap_realize_pipeline(self._pipeline_obj, len(outs),c_outs)

@extclass(HIR.Pipeline)
def __call__(self,params,imgs,outputs):
    if not hasattr(self,'_executor'):
        self._executor = _HIR_JIT_Execution(self)
    self._executor._exec(params,imgs,outputs)
del __call__

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#
#           Below dumps HIR to a C++ string and JIT compiles it
#
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Helpers and directory setup

def _shell(cstr):
  subprocess.run(cstr, check=True, shell=True)

_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_C_CACHE        = os.path.join(_HERE_DIR,'.atl_halide_cache')
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
#   Lowering HIR to a string

class HIR_CPP_String:
    def __init__(self, name, pipe, param_vals, img_ranges, out_ranges):
        assert type(pipe) is HIR.Pipeline

        self._ctxt      = ChainMap({})
        self._names     = ChainMap({})
        self._tab       = ""
        self._lines     = []

        # start of file...
        self.line('#include "Halide.h"')
        self.line('using namespace Halide;')
        self.line('')
        self.line(f"class Generator_for_{name} : "+
                  f"public Halide::Generator<Generator_for_{name}>"+"{")
        self.line("public:")
        # entering class body
        self.push_scope(name=False)

        # inputs
        # param inputs
        for p in pipe.params:
            ctyp = self.hir_to_ctype(p.typ)
            name = self.new_name(p.name)
            self.line(f'Input<{ctyp}> {name}{{"{name}"}}')
        # img inputs
        for img in pipe.imgs:
            ctyp = self.hir_to_ctype(img.typ)
            name = self.new_name(img.name)
            order = img.ndim
            self.line(f'Input<Buffer<{ctyp}>> {name}{{"{name}", {order}}};')
        self.line("")

        # outputs
        # output funcs
        for f in pipe.outputs:
            assert type(f) is HIR.Func
            ctyp = self.hir_to_ctype(f.typ)
            name = self.new_name(f.name)
            order = f.ndim
            self.line(f'Output<Buffer<{ctyp}>> {name}{{"{name}", {f.ndim}}};')
        self.line("")

        # void generate
        self.line("void generate() {")
        self.push_scope()
        for s in pipe.stmts:
            self.compile_stmt(s)
        self.pop_scope()
        self.line("}")
        self.line("")

        # void schedule
        self.line("void schedule() {")
        self.push_scope(name=False)
        self.line("if(auto_schedule) {")
        self.push_scope()
        # estimate input params
        for p,val in zip(pipe.params,param_vals):
            self.line(f'{self._ctxt[p.name]}.set_estimate({val});')
        self.line("")

        # estimate input imgs
        for img, shape in zip(pipe.imgs, img_ranges):
            name    = self._ctxt[img.name]
            rngs    = [ f"{{0,{s}}}" for s in shape ]
            rngs    = "{"+(",".join(rngs))+"}"
            self.line(f'{name}.set_estimates({rngs});')
        self.line("")

        # estimate outputs
        for f, shape in zip(pipe.outputs, out_ranges):
            name    = self._ctxt[f.name]
            rngs    = [ f"{{0,{s}}}" for s in shape ]
            rngs    = "{"+(",".join(rngs))+"}"
            self.line(f'{name}.set_estimates({rngs});')

        self.pop_scope()
        self.line("}")
        self.pop_scope(name=False)
        self.line("}")

        # exiting class body
        self.pop_scope(name=False)
        self.line("};")
        # bottom of file...
        self.line(f"HALIDE_REGISTER_GENERATOR(Generator_for_{name}, {name});")

    def result(self):
        return '\n'.join(self._lines)

    def compile_stmt(self, stmt):
        # determine whether func needs to be declared
        if type(stmt) is HIR.PureDef:
            if stmt.f.name in self._ctxt:
                assert any( stmt.f.name == o.name for o in self._pipe.outputs )
            else:
                self.line(f"Func {self.new_name(stmt.f.name)};")

        self.push_scope(tab=False)
        # determine variables that need to be declared
        if type(stmt) is HIR.PureDef:
            vars = stmt.args
        elif type(stmt) is HIR.Update:
            vars = set()
            for a in e.args:
                if type(a) is HIR.Evar:
                    vars.add(a.v.name)
            vars = list(vars)
        for v in vars:
            self.line(f"Var {self.new_name(v)};")

        # determine rdoms that need to be declared
        rdoms   = {}
        if type(stmt) is HIR.Update:
            for a in stmt.args:
                self.rdoms_used(a, rdoms)
        self.rdoms_used(stmt.rhs, rdoms)
        for r in rdoms:
            bds     = rdoms[r]
            bdargs  = []
            for rng in bds:
                bdargs.append( self.compile_expr(rng.lo) )
                bdargs.append( self.compile_expr(rng.extent) )
            self.line(f"RDom {self.new_name(r)}({','.join(bdargs)});")

        # next, build the left-hand-side of the stmt
        fname       = self._ctxt[stmt.f.name]
        assert len(stmt.args) > 0
        if type(stmt) is HIR.PureDef:
            args    = [ self._ctxt[a.name] for a in stmt.args ]
        else:
            assert type(stmt) is HIR.Update
            args    = [ self.compile_expr(a) for a in stmt.args ]
        rhs         = self.compile_expr(stmt.rhs)
        self.line(f"{fname}[{','.join(args)}] = {rhs};")

        self.pop_scope(tab=False)

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
    def compile_expr(self, e, prec=0):
        etyp = type(e)

        if etyp is HIR.Const:
            return str(e.v)
        elif etyp is HIR.Evar:
            return self._ctxt[e.v.name]
        elif etyp is HIR.Erdom:
            return self._ctxt[e.r.name]
        elif etyp is HIR.Eparam:
            return self._ctxt[e.p.name]
        elif etyp is HIR.BinOp:
            op_prec = HIR_CPP_String._prec[e.op]
            lhs = self.compile_expr(e.lhs, prec=op_prec)
            rhs = self.compile_expr(e.rhs, prec=op_prec+1)
            exp = f'{lhs} {e.op} {rhs}'
            if prec > op_prec:
                exp = f'({exp})'
            return exp
        elif etyp is HIR.Min or etyp is HIR.Max:
            op  = "min" if etyp is HIR.Min else "max"
            lhs = self.compile_expr(e.lhs)
            rhs = self.compile_expr(e.rhs)
            return f"{op}({lhs}, {rhs})"
        elif etyp is HIR.MathFn1:
            return f'{e.name}({self.compile_expr(e.arg)})'
        elif etyp is HIR.Clamp:
            val = self.compile_expr(e.val)
            lo = self.compile_expr(e.lo)
            hi = self.compile_expr(e.hi)
            return f'clamp({val}, {lo}, {hi})'
        elif etyp is HIR.Pow:
            base = self.compile_expr(e.base)
            exp = self.compile_expr(e.exp)
            return f'pow({base}, {exp})'
        elif etyp is HIR.ATan2:
            y = self.compile_expr(e.y)
            x = self.compile_expr(e.x)
            return f'atan2({y}, {x})'
        elif etyp is HIR.Select:
            pred = self.compile_expr(e.pred)
            lhs = self.compile_expr(e.lhs)
            rhs = self.compile_expr(e.rhs)
            return f'select({pred}, {lhs}, {rhs})'
        elif etyp is HIR.FAccess:
            nm = e.f.name if type(e.f) is HIR.Func else e.f.img.name
            name = self._ctxt[nm]
            args = [ self.compile_expr(a) for a in e.args ]
            return f'{name}({",".join(args)})'
        elif etyp is HIR.BigSum:
            r    = self._ctxt[e.r.name]
            body = self.compile_expr(e.body)
            return f'sum({r}, {body})'
        else: assert False, "bad case"

    def line(self, s):
        self._lines.append(f"{self._tab}{s}")

    def push_scope(self, name=True, tab=True):
        if tab:
          self._tab  += "  "
        if name:
          self._ctxt  = self._ctxt.new_child()
          self._names = self._names.new_child()

    def pop_scope(self, name=True, tab=True):
        if tab:
            self._tab   = self._tab[0:-2] # trim two chars off the end
        if name:
            self._ctxt  = self._ctxt.parents
            self._names = self._names.parents

    _typ_convert_tbl = {
        "f32":  "float",
        "f64":  "double",
        "u8":   "uint8_t",
        "u16":  "uint16_t",
        "u32":  "uint32_t",
        "u64":  "uint64_t",
        "i8":   "int8_t",
        "i16":  "int16_t",
        "i32":  "int32_t",
        "i64":  "int64_t",
    }
    def hir_to_ctype(typ):
        assert type(typ) is HIR.Type
        assert typ.lanes == 1
        return HIR_CPP_String._typ_convert_tbl[typ.base]

    def rdoms_used(self, e, rdom_bds):
        etyp = type(e)

        if etyp is HIR.Erdom:
            if e.r.name not in rdom_bds:
                rdom_bds[e.r.name] = e.r.bounds
        elif (etyp is HIR.BinOp or
              etyp is HIR.Min or etyp is HIR.Max):
            self.rdoms_used(e.lhs, rdom_bds)
            self.rdoms_used(e.rhs, rdom_bds)
        elif etyp is HIR.MathFn1:
            self.rdoms_used(e.arg, rdom_bds)
        elif etyp is HIR.Clamp:
            self.rdoms_used(e.val, rdom_bds)
            self.rdoms_used(e.lo, rdom_bds)
            self.rdoms_used(e.hi, rdom_bds)
        elif etyp is HIR.Pow:
            self.rdoms_used(e.base, rdom_bds)
            self.rdoms_used(e.exp, rdom_bds)
        elif etyp is HIR.ATan2:
            self.rdoms_used(e.x, rdom_bds)
            self.rdoms_used(e.y, rdom_bds)
        elif etyp is HIR.Select:
            self.rdoms_used(e.pred, rdom_bds)
            self.rdoms_used(e.lhs, rdom_bds)
            self.rdoms_used(e.rhs, rdom_bds)
        elif etyp is HIR.FAccess:
            for a in e.args:
                self.rdoms_used(a, rdom_bds)
        elif etyp is HIR.BigSum:
            if e.r.name not in rdom_bds:
                rdom_bds[e.r.name] = e.r.bounds
            self.rdoms_used(e.body, rdom_bds)

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
            # iterate the following process till we are sure
            # we have a fresh name
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


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   JIT compilation of generated CPP files


class Halide_CJit:
  """ Manage JIT compilation of ATL -> CPP/Halide code
  """

  def __init__(self, name, pipe):
    assert type(pipe) is HIR.Pipeline

    self._pipe    = pipe

    # run once just to compute hash values...
    hstr, cstr    = HIR_CPP_String(_C_CACHE,
                                   f"{name}.h", f"{name}.cpp",
                                   pipe).result()
    hashstr       = hashlib.md5(cstr.encode('utf-8')).hexdigest()

    fname         = name + hashstr
    h_filename    = os.path.join(_C_CACHE,f"{fname}.h")
    c_filename    = os.path.join(_C_CACHE,f"{fname}.cpp")
    so_filename   = os.path.join(_C_CACHE,f"{fname}.so")
    comp_cmd      = (f"clang++ -Wall -Werror -fPIC -O3 -shared -std=c++11 "
                     f"-I {_C_CACHE} "
                     f"-o {so_filename} {c_filename}")
    hstr, cstr    = HIR_CPP_String(_C_CACHE,
                                   f"{fname}.h", f"{fname}.cpp",
                                   pipe).result()

    #print(hstr)
    #print(cstr)

    def matches_file(src, fname):
      if not os.path.isfile(fname):
        return False
      else:
        with open(fname, 'r', encoding = 'utf-8') as F:
          return F.read() == src
    def write_file(src, fname):
      with open(fname, 'w', encoding = 'utf-8') as F:
        F.write(src)

    # do we need to rebuild the corresponding SO?
    if (not matches_file(hstr, h_filename) or
        not matches_file(cstr, c_filename)):
      write_file(hstr, h_filename)
      write_file(cstr, c_filename)
      #print(comp_cmd)
      _shell(comp_cmd)

    # load the module regardless
    module        = ctypes.CDLL(so_filename)
    self._module  = module
    self._cfun    = getattr(module, name)

    #wrap the module function with types
    raise NotImplementedError("TODO: DECIDE TYPING")
    atyps = []
    for sz in self._proc.sizes:
      atyps.append(get_ctype(int))
    for a in self._proc.args:
      if str(a.name) == "output":
        atyps.append(ctypes.POINTER(get_ctype(a.type)))
      else:
        atyps.append(get_ctype(a.type))
    for sz in self._proc.relargs:
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

    pack_output(out_obj, out, self._func.rettype)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
