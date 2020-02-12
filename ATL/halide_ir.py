
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
    func   = Func      ( string name )
           | ImgFunc   ( img img )
    
    expr   = Const     ( object v, type typ )
           | Evar      ( var v )
           | Erdom     ( rdom r )
           | Eparam    ( param p )
           | BinOp     ( op op, expr lhs, expr rhs )
           | MathFn1   ( string name, expr arg )
           | Min       ( expr lhs, expr rhs )
           | Max       ( expr lhs, expr rhs )
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
        
    def _exec(self, params, imgs, outputs):
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



