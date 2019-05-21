import ctypes
import os
import sys
import subprocess

def _shell(cstr):
    subprocess.run(cstr, check=True, shell=True)

# Declare where the halide distribution is at
_HALIDE_PATH    = os.path.expanduser('~/install/halide')

# Derive the Halide build locations
_HALIDE_LIB     = os.path.join(_HALIDE_PATH,'lib')
_HALIDE_INC     = os.path.join(_HALIDE_PATH,'include')
_HALIDE_STATIC  = os.path.join(_HALIDE_LIB,'libHalide.a')

# Make sure that the Halide library is actually there
if not os.path.isfile(_HALIDE_STATIC):
    raise IOError(f"Halide library not found at {HALIDE_STATIC}")

# Make sure that we have a cache directory
_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_C_DIR          = os.path.join(_HERE_DIR,'._halide_c_wrap_cache')
if not os.path.isdir(_C_DIR):
    os.mkdir(_C_DIR)

# Make sure we have a way to blow away the cache
def _clean_cache():
    _shell( 'rm ' + os.path.join(_C_DIR,"*") )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Basic Type Wrapping
_CTYPES_PTR_TYPE    = type(ctypes.POINTER(ctypes.c_int))
_CTYPES_SIMPLE_TYPE = type(ctypes.c_int)

_str_to_pair = {
    "b":    {'ct':ctypes.c_bool,'s':'_Bool'},
    "i8":   {'ct':ctypes.c_byte,'s':'int8_t'},
    "i16":  {'ct':ctypes.c_short,'s':'int16_t'},
    "i32":  {'ct':ctypes.c_int,'s':'int32_t'},
    "i64":  {'ct':ctypes.c_longlong,'s':'int64_t'},
    "u8":   {'ct':ctypes.c_ubyte,'s':'uint8_t'},
    "u16":  {'ct':ctypes.c_ushort,'s':'uint16_t'},
    "u32":  {'ct':ctypes.c_uint,'s':'uint32_t'},
    "u64":  {'ct':ctypes.c_ulonglong,'s':'uint64_t'},
    "f32":  {'ct':ctypes.c_float,'s':'float'},
    "f64":  {'ct':ctypes.c_double,'s':'double'},
    "v":    {'ct':None,'s':'void'},
    "vp":   {'ct':ctypes.c_void_p,'s':'void *'},
    "vpp":  {'ct':ctypes.POINTER(ctypes.c_void_p),'s':'void **'},
    "s":    {'ct':ctypes.c_char_p,'s':'char *'},
}
_typ_to_pair = {}
for _,pair in _str_to_pair.items():
    _typ_to_pair[pair['ct']] = pair
del pair

def wrapT(sig):
    if type(sig) is str:
        if sig in _str_to_pair:
            return _str_to_pair[sig]
        else:
            raise TypeError(f"unrecognized C type string: {sig}")
            
    # convert our defined object types into type pairs
    if issubclass(sig,ctypes.Structure):
        return {'ct':sig,'s':sig._c_type_str}
    
    # lift simple types up to type pairs
    elif type(sig) == _CTYPES_SIMPLE_TYPE:
        if sig in _typ_to_pair:
            return _typ_to_pair[sig]
        else:
            raise TypeError(f"unsupported C type: {sig}")
    
    # handle pointer types
    elif type(sig) == _CTYPES_PTR_TYPE:
        sub = wrapT(sig._type_)
        return {'ct':sig,'s':sub['s']+" *"}
    
    else:
        raise TypeError(f"unrecognized argument type: {type(sig)}")

# --------------------------------------------------------------------------- #

class HWGen:
    _name           = "C_Wrap_Halide"
    _inc_str        = (
    '#include "Halide.h"\n'+
    '#include <stdio.h>\n'+
    '#include <stdlib.h>\n'+
    '#include <stdint.h>\n'
    )

    _decl_strs      = []
    _defn_strs      = []
    _ctype_wraps    = []
    _module         = None

    def c_decl(s):
        HWGen._decl_strs.append(s)
    def cpp_defn(s):
        HWGen._defn_strs.append(s)
    def on_gen(f):
        HWGen._ctype_wraps.append(f)

    def fun(name,args,ret,body):
        astr = ', '.join([ wrapT(a[1])['s'] +' '+a[0]
                           for a in args
                         ])
        rstr = wrapT(ret)['s']
        
        atyps = [ wrapT(a[1])['ct'] for a in args ]
        rtyp  = wrapT(ret)['ct']
        
        HWGen._decl_strs.append(f"{rstr} {name}({astr});")
        HWGen._defn_strs.append(f"{rstr} {name}({astr}) {{{body}\n}}")
        def wrap(mod):
            f = getattr(mod,name)
            f.argtypes = atyps
            f.restype  = rtyp
        HWGen._ctype_wraps.append(wrap)

    def gen():
        #global _module
        all_decls   = '\n'.join(HWGen._decl_strs)
        all_defns   = '\n\n'.join(HWGen._defn_strs)
        src_str     =  (f'{HWGen._inc_str}\n\n'
                        f'extern "C" {{\n'
                        f'{all_decls}\n'
                        f'}}\n\n'
                        f'{all_defns}\n')
        
        CPP     = os.path.join(_C_DIR, f"{HWGen._name}.cpp")
        SO      = os.path.join(_C_DIR, f"lib{HWGen._name}.so")
        
        # first, get timestamps on needed resources
        # we'll use these to conditionally compile
        def get_time(s):
            if not os.path.exists(s):
                return None
            else:
                return os.path.getmtime(s)
        
        cpp_time    = get_time(CPP)
        so_time     = get_time(SO)
        h_time      = get_time(_HALIDE_STATIC)
        
        # Check whether the CPP file needs to be re-written
        write_cpp   = True if cpp_time == None else False
        if cpp_time:
            with open(CPP,'r',encoding = 'utf-8') as f:
                if src_str != f.read():
                    write_cpp = True
        # possibly rewrite the CPP file
        if write_cpp:
            if so_time: cpp_time = so_time + 1
            with open(CPP,'w',encoding = 'utf-8') as f:
                f.write(src_str)
        
        # Check whether the SO needs to be re-compiled
        if (not cpp_time or not so_time or
            so_time < cpp_time or
            so_time < h_time):
                cmd = (f"clang++ -Wall -Werror -fPIC -O3 -shared -std=c++11 "
                       f"-I {_HALIDE_INC} {_HALIDE_STATIC} -lz "
                       f"-o {SO} {CPP}")
                print(cmd)
                _shell(cmd)
                if HWGen._module != None:
                    raise IOError(f"library {_name} already loaded")
        
        # Load the module if needed
        if HWGen._module == None:
            HWGen._module = ctypes.CDLL(SO)
            for wrap in HWGen._ctype_wraps:
                wrap(HWGen._module)
        
        return HWGen._module

# --------------------------------------------------------------------------- #

class halide_type_t(ctypes.Structure):
    _c_type_str = "struct halide_type_t"
    _fields_ = [
        ('code',  ctypes.c_ubyte),
        ('bits',  ctypes.c_ubyte),
        ('lanes', ctypes.c_ushort),
    ]

class halide_dimension_t(ctypes.Structure):
    _c_type_str = "struct halide_dimension_t"
    _fields_ = [
        ('min',    ctypes.c_int),
        ('extent', ctypes.c_int),
        ('stride', ctypes.c_int),
        ('flags',  ctypes.c_uint),
    ]

class halide_buffer_t(ctypes.Structure):
    _c_type_str = "struct halide_buffer_t"
    _fields_ = [
        ('device',            ctypes.c_ulonglong),
        ('device_interface',  ctypes.c_void_p),
        ('host',              ctypes.POINTER(ctypes.c_ubyte)),
        ('flags',             ctypes.c_ulonglong),
        ('type',              halide_type_t),
        ('dimensions',        ctypes.c_int),
        ('dim',               ctypes.POINTER(halide_dimension_t)),
        ('padding',           ctypes.c_void_p),
    ]

# --------------------------------------------------------------------------- #

for C,ctyp,htyp in [
    ('E','struct hw_expr_t', 'Halide::Expr'),
    ('V','struct hw_var_t',  'Halide::Var' ),
    ('R','struct hw_rdom_t', 'Halide::RDom'),
    ('F','struct hw_func_t', 'Halide::Func'),
    ('I','struct hw_img_t',  'Halide::ImageParam'),
    ('P','struct hw_param_t','Halide::Param<>'),
    ]:
        HWGen.c_decl(f"{ctyp} {{"
        f"    uint64_t id;"
        f"}};")
        HWGen.cpp_defn(f"{ctyp} _to_{C}({htyp} * x) {{"
        f"    return ({ctyp}){{uint64_t(x)}};"
        f"}}")
        HWGen.cpp_defn(f"{htyp} * _from_{C}({ctyp} x) {{"
        f"    return ({htyp} *)(x.id);"
        f"}}")
del C,ctyp,htyp


# EXPR
class hw_expr_t(ctypes.Structure):
    _c_type_str = "struct hw_expr_t"
    _fields_ = [ ('id', ctypes.c_ulonglong) ]

# VAR
class hw_var_t(ctypes.Structure):
    _c_type_str = "struct hw_var_t"
    _fields_ = [ ('id', ctypes.c_ulonglong) ]

# RDOM
class hw_rdom_t(ctypes.Structure):
    _c_type_str = "struct hw_rdom_t"
    _fields_ = [ ('id', ctypes.c_ulonglong) ]

# FUNC
class hw_func_t(ctypes.Structure):
    _c_type_str = "struct hw_func_t"
    _fields_ = [ ('id', ctypes.c_ulonglong) ]

# IMG
class hw_img_t(ctypes.Structure):
    _c_type_str = "struct hw_img_t"
    _fields_ = [ ('id', ctypes.c_ulonglong) ]

# PARAM
class hw_param_t(ctypes.Structure):
    _c_type_str = "struct hw_param_t"
    _fields_ = [ ('id', ctypes.c_ulonglong) ]

# --------------------------------------------------------------------------- #

_H_v01_constlist = [
    # device constants
    ('u64','DeviceAPI_None',            'Halide::DeviceAPI::None'),
    ('u64','DeviceAPI_Host',            'Halide::DeviceAPI::Host'),
    ('u64','DeviceAPI_Default_GPU',     'Halide::DeviceAPI::Default_GPU'),
    # type constants
    ('u8', 'type_int',                  'halide_type_int'),
    ('u8', 'type_uint',                 'halide_type_uint'),
    ('u8', 'type_float',                'halide_type_float'),
    ('u8', 'type_handle',               'halide_type_handle'),
]

def _add_const(typ, nm, c_nm):
    typ_str = wrapT(typ)['s']
    HWGen.fun(f"hwrap_get_{nm}",[],typ,
              f"return ({typ_str})({c_nm});")
for typ,nm,c_nm in _H_v01_constlist:
    _add_const(typ,nm,c_nm)
del typ,nm,c_nm

def _unpack_consts(mod):
    for typ,nm,c_nm in _H_v01_constlist:
        fn = getattr(mod,f"hwrap_get_{nm}")
        setattr(mod,nm,fn())
        # erase the function once we've used it.
        setattr(mod,f"hwrap_get_{nm}",None)
# wait until the module is loaded to unpack constants
HWGen.on_gen(_unpack_consts)

# --------------------------------------------------------------------------- #
# Destructors
HWGen.fun("hwrap_delete_func", [('f_handle',hw_func_t)],'v',"""
    delete _from_F(f_handle);""")
HWGen.fun("hwrap_delete_var",[('v_handle',hw_var_t)],'v',"""
    delete _from_V(v_handle);""")
HWGen.fun("hwrap_delete_rdom",[('r_handle',hw_rdom_t)],'v',"""
    delete _from_R(r_handle);""")
HWGen.fun("hwrap_delete_expr",[('e_handle',hw_expr_t)],'v',"""
    delete _from_E(e_handle);""")
HWGen.fun("hwrap_delete_img",[('i_handle',hw_img_t)],'v',"""
    delete _from_I(i_handle);""")
HWGen.fun("hwrap_delete_param",[('p_handle',hw_param_t)],'v',"""
    delete _from_P(p_handle);""")

# DESTRUCTORS for Func, Expr, Var, RDom...
def _install_destructors(mod):
    def _hw_func_t_del(self):
        if self.id != 0:
            mod.hwrap_delete_func(self)
            self.id = 0
    def _hw_expr_t_del(self):
        if self.id != 0:
            mod.hwrap_delete_expr(self)
            self.id = 0
    def _hw_var_t_del(self):
        if self.id != 0:
            mod.hwrap_delete_var(self)
            self.id = 0
    def _hw_rdom_t_del(self):
        if self.id != 0:
            mod.hwrap_delete_rdom(self)
            self.id = 0
    def _hw_img_t_del(self):
        if self.id != 0:
            mod.hwrap_delete_img(self)
            self.id = 0
    def _hw_param_t_del(self):
        if self.id != 0:
            mod.hwrap_delete_param(self)
            self.id = 0
        
    hw_func_t.__del__  = _hw_func_t_del
    hw_expr_t.__del__  = _hw_expr_t_del
    hw_var_t.__del__   = _hw_var_t_del
    hw_rdom_t.__del__  = _hw_rdom_t_del
    hw_img_t.__del__   = _hw_img_t_del
    hw_param_t.__del__ = _hw_param_t_del
    
# install destructors onto objects
HWGen.on_gen(_install_destructors)


# FUNC
HWGen.fun("hwrap_new_func",[('name','s')],hw_func_t,"""
    return _to_F(new Halide::Func(name));""")
HWGen.fun("hwrap_set_func_bound_estimate",
    [('f',hw_func_t),('d','i32'),
     ('min',hw_expr_t),('extent',hw_expr_t)],'v',"""
    auto args = _from_F(f)->args();
    _from_F(f)->estimate(args[d],*_from_E(min),*_from_E(extent));""")

# VAR
HWGen.fun("hwrap_new_var",[('name','s')],hw_var_t,"""
    return _to_V(new Halide::Var(name));""")

# RDOM
HWGen.fun("hwrap_new_rdom",
    [('name','s'),
     ('n_dim','i32'),
     ('ranges',ctypes.POINTER(hw_expr_t)),],
    hw_rdom_t,"""
    std::vector< std::pair< Halide::Expr, Halide::Expr > > r;
    for(int k=0; k<n_dim; k++)
        r.push_back(std::make_pair( *_from_E(ranges[2*k]),
                                    *_from_E(ranges[2*k+1]) ));
    return _to_R(new Halide::RDom(r,name));""")

# PARAM
HWGen.fun("hwrap_new_param",
    [('name','s'),
     ('typ',halide_type_t),], hw_param_t,"""
     return _to_P(new Halide::Param<>(Halide::Type(typ), name));""")
HWGen.fun("hwrap_set_param_range",
    [('param',hw_param_t),('lo',hw_expr_t),('hi',hw_expr_t)],'v',"""
    _from_P(param)->set_range(*_from_E(lo),*_from_E(hi));""")
HWGen.fun("hwrap_set_param_estimate",
    [('param',hw_param_t),('e',hw_expr_t)],'v',"""
    _from_P(param)->parameter().set_estimate(*_from_E(e));""")
HWGen.fun("hwrap_set_param",
    [('param',hw_param_t),('val','vp')],'v',"""
    Halide::Type typ = _from_P(param)->type();
    auto v = (halide_scalar_value_t*)(val);
    _from_P(param)->parameter().set_scalar(typ,*v);""")

# IMG
HWGen.fun("hwrap_new_img",
    [('name','s'),
     ('n_dim','i32'),
     ('typ',halide_type_t),], hw_img_t,"""
    return _to_I(new Halide::ImageParam(Halide::Type(typ), n_dim, name));""")
HWGen.fun("hwrap_set_img_bound_estimate",
    [('img',hw_img_t),('d','i32'),('min',hw_expr_t),('extent',hw_expr_t)],'v',"""
    _from_I(img)->dim(d).set_bounds_estimate(*_from_E(min),*_from_E(extent));""")
HWGen.fun("hwrap_set_img",
    [('img',hw_img_t),('input',ctypes.POINTER(halide_buffer_t))],'v',"""
    _from_I(img)->set(Halide::Buffer<>(*input));""")
HWGen.fun("hwrap_img_to_func",[('i_handle',hw_img_t)],hw_func_t,"""
    return _to_F(new Halide::Func( _from_I(i_handle)->in() ));""")

# EXPR
# convert constant values to Expr
for typstr in ['i8','i16','i32','i64','u8','u16','u32','u64','f32','f64']:
    HWGen.fun(f"hwrap_{typstr}_to_expr",[('c',typstr)],hw_expr_t,"""
        return _to_E(new Halide::Expr( c ));""")
del typstr
# converts a Var to a Int32-type Expr
HWGen.fun("hwrap_var_to_expr",[('v_handle',hw_var_t)],hw_expr_t,"""
    return _to_E(new Halide::Expr( *(_from_V(v_handle)) ));""")
# converts an RDom to a Int32-type Expr
HWGen.fun("hwrap_rdom_to_expr",[('r_handle',hw_rdom_t)],hw_expr_t,"""
    return _to_E(new Halide::Expr( *(_from_R(r_handle)) ));""")
# converts a Param to an Expr
HWGen.fun("hwrap_param_to_expr",[('p_handle',hw_param_t)],hw_expr_t,"""
    return _to_E(new Halide::Expr( *(_from_P(p_handle)) ));""")
# binary operations
for opnm,binop in [
    ('add','+'),
    ('sub','-'),
    ('mul','*'),
    ('div','/'),
    ('eq', '=='),]:
        HWGen.fun(f"hwrap_{opnm}",
            [('lhs',hw_expr_t),('rhs',hw_expr_t)],hw_expr_t,"\n"+
            "return _to_E(new Halide::Expr("+
                f" (*_from_E(lhs)) {binop} (*_from_E(rhs)) ));")
#HWGen.fun("hwrap_add",[('lhs',hw_expr_t),('rhs',hw_expr_t)],hw_expr_t,"""
#    return _to_E(new Halide::Expr(  *_from_E(lhs)  +  *_from_E(rhs) ));""")
#HWGen.fun("hwrap_sub",[('lhs',hw_expr_t),('rhs',hw_expr_t)],hw_expr_t,"""
#    return _to_E(new Halide::Expr(  *_from_E(lhs)  -  *_from_E(rhs) ));""")
#HWGen.fun("hwrap_mul",[('lhs',hw_expr_t),('rhs',hw_expr_t)],hw_expr_t,"""
#    return _to_E(new Halide::Expr( (*_from_E(lhs)) * (*_from_E(rhs)) ));""")
#HWGen.fun("hwrap_div",[('lhs',hw_expr_t),('rhs',hw_expr_t)],hw_expr_t,"""
#    return _to_E(new Halide::Expr( (*_from_E(lhs)) / (*_from_E(rhs)) ));""")
#HWGen.fun("hwrap_eq",[('lhs',hw_expr_t),('rhs',hw_expr_t)],hw_expr_t,"""
#    return _to_E(new Halide::Expr( (*_from_E(lhs)) == (*_from_E(rhs)) ));""")
HWGen.fun("hwrap_select",[('cond',hw_expr_t),
                          ('if_T',hw_expr_t),
                          ('if_F',hw_expr_t)],hw_expr_t,"""
    return _to_E(new Halide::Expr(Halide::select(
        *_from_E(cond), *_from_E(if_T), *_from_E(if_F) )));""")
# func access
HWGen.fun("hwrap_access_func",
    [('f',hw_func_t),
     ('n_idx','i32'),
     ('idx',ctypes.POINTER(hw_expr_t))], hw_expr_t,"""
    std::vector<Halide::Expr> args;
    for(int k=0; k<n_idx; k++)
        args.push_back( *_from_E(idx[k]) );
    Halide::FuncRef fr = (*_from_F(f))(args);
    return _to_E(new Halide::Expr( fr ));""")

# big sum
HWGen.fun("hwrap_big_sum",[('r',hw_rdom_t),('e',hw_expr_t)],hw_expr_t,"""
    return _to_E(new Halide::Expr(Halide::sum( *_from_R(r),
                                               *_from_E(e) )));""")

# Statements
HWGen.fun("hwrap_pure_def",
    [('fh',hw_func_t),
     ('n_idx','i32'),
     ('idx',ctypes.POINTER(hw_var_t)),
     ('rhs',hw_expr_t)], "v","""
    std::vector<Halide::Var> args;
    for(int k=0; k<n_idx; k++)
        args.push_back( *_from_V(idx[k]) );
    (*_from_F(fh))(args) = *_from_E(rhs);""")
HWGen.fun("hwrap_update",
    [('fh',hw_func_t),
     ('n_idx','i32'),
     ('idx',ctypes.POINTER(hw_expr_t)),
     ('rhs',hw_expr_t)], "v","""
    std::vector<Halide::Expr> args;
    for(int k=0; k<n_idx; k++)
        args.push_back( *_from_E(idx[k]) );
    (*_from_F(fh))(args) = *_from_E(rhs);""")


# DEVICE INTERFACE
HWGen.fun("hwrap_get_jit_device", [('_d','u64')],'vp', """
    Halide::DeviceAPI d = (Halide::DeviceAPI)(_d);
    return (void *)(Halide::get_device_interface_for_device_api(d));""")

# Realizing a result (with JiT compilation)
HWGen.fun("hwrap_realize_func",
    [('self',hw_func_t),('output',ctypes.POINTER(halide_buffer_t))],'v',"""
    Halide::Buffer<> buf(*output);
    _from_F(self)->realize(Halide::Realization(buf),
                           Halide::get_host_target());""")

HWGen.fun("hwrap_autoschedule_func",[('f',hw_func_t)],'v',"""
    std::vector<Halide::Internal::Function> fs;
    fs.push_back(_from_F(f)->function());
    auto s = Halide::Internal::generate_schedules(
                fs,
                Halide::get_host_target(),
                Halide::MachineParams::generic());""")

HWGen.fun("hwrap_func_print_loop_nest",[('f',hw_func_t)],'v',"""
    _from_F(f)->print_loop_nest();""")

# --------------------------------------------------------------------------- #

C = HWGen.gen()


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cleanup the Namespace

del _shell
del _HALIDE_PATH
del _HALIDE_LIB
del _HALIDE_INC
del _HALIDE_STATIC
del _HERE_DIR
del _C_DIR
del _clean_cache

# --------------------------------------------------------------------------- #

del _CTYPES_PTR_TYPE
del _CTYPES_SIMPLE_TYPE

del _str_to_pair
del _typ_to_pair
del wrapT

# --------------------------------------------------------------------------- #

del HWGen

# --------------------------------------------------------------------------- #

del _H_v01_constlist
del _add_const
del _unpack_consts
del _install_destructors


del os, sys, subprocess

