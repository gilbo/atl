# See "How-to Represent IRs.ipynb" for information about
# the construction of this file.

import asdl
from types import ModuleType
from weakref import WeakValueDictionary

def _asdl_parse(str):
    parser = asdl.ASDLParser()
    module = parser.parse(str)
    return module

def _build_superclasses(asdl_mod):
    scs = {}
    def create_invalid_init(nm):
        def invalid_init(self):
            assert false, f"{nm} should never be instantiated"
        return invalid_init
    
    for nm,v in asdl_mod.types.items():
        if isinstance(v,asdl.Sum):
            scs[nm] = type(nm,(),{"__init__" : create_invalid_init(nm)})
        elif isinstance(v,asdl.Product):
            scs[nm] = type(nm,(),{})
    return scs

_builtin_checks = {
    'string'  : lambda x: type(x) is str,
    'int'     : lambda x: type(x) is int,
    'object'  : lambda x: x is not None,
    'float'   : lambda x: type(x) is float,
    'bool'    : lambda x: type(x) is bool,
}

def _build_checks(asdl_mod, scs, ext_checks):
    checks = _builtin_checks.copy()
    def make_check(sc):
        return lambda x: isinstance(x,sc)
    
    for nm in ext_checks:
        checks[nm] = ext_checks[nm]
    for nm in scs:
        assert not nm in checks, f"Name conflict for type '{nm}'"
        sc = scs[nm]
        checks[nm] = make_check(sc)
    return checks

def _build_classes(asdl_mod, ext_checks={}):
    SC   = _build_superclasses(asdl_mod)
    CHK  = _build_checks(asdl_mod, SC, ext_checks)
    
    mod  = ModuleType(asdl_mod.name)
    
    Err  = type(asdl_mod.name+"Err",(Exception,),{})
    
    def basic_check(i,name,typ,indent="    "):
        typname = typ
        if typ in SC:
            typname = asdl_mod.name + "." + typ
        return (f"{indent}if not CHK['{typ}']({name}):\n"
                f"{indent}    raise Err("
                f"'expected arg {i} \"{name}\" "
                f"to be type \"{typname}\"')")
    def opt_check(i,name,typ,indent="    "):
        subidnt = indent + '    '
        return (f"{indent}if {name} is not None:\n"
                f"{basic_check(i,name,typ,subidnt)}")
    def seq_check(i,name,typ,indent="    "):
        subidnt = indent + '        '
        return (f"{indent}if type({name}) is list:\n"
                f"{indent}    for j,e in enumerate({name}):\n"
                f"{basic_check(i,name+'[j]',typ,subidnt)}")
    
    def create_initfn(C_name, fields):
        argstr   = ', '.join([ f.name for f in fields ])
        checks   = '\n'.join([
            seq_check(i,f.name,f.type) if f.seq else
            opt_check(i,f.name,f.type) if f.opt else
            basic_check(i,f.name,f.type)
            for i,f in enumerate(fields)
        ])
        assign   = '\n    '.join([
            f"self.{f.name} = {f.name}"
            for f in fields
        ])
        if len(fields) == 0:
            checks = "    pass"
            assign = "pass"
        
        exec_out = { 'Err': Err, 'CHK': CHK }
        exec_str = (f"def {C_name}_init(self,{argstr}):"
                    f"\n{checks}"
                    f"\n    {assign}")
        # un-comment this line to see what's
        # really going on
        #print(exec_str)
        exec(exec_str, exec_out)
        return exec_out[C_name + '_init']
    
    def create_reprfn(C_name, fields):
        prints   = ','.join([
            f"{f.name}={{self.{f.name}}}"
            for f in fields
        ])
        exec_out = { 'Err': Err }
        exec_str = (f"def {C_name}_repr(self):"
                    f"\n    return f\"{C_name}({prints})\"")
        # un-comment this line to see what's
        # really going on
        #print(exec_str)
        exec(exec_str, exec_out)
        return exec_out[C_name + '_repr']
        
    def create_prod(nm,t):
        C          = SC[nm]
        fields     = t.fields
        C.__init__ = create_initfn(nm,fields)
        C.__repr__ = create_reprfn(nm,fields)
        return C
    
    def create_sum_constructor(tname,cname,T,fields):
        C          = type(cname,(T,),{
            '__init__' : create_initfn(cname,fields),
            '__repr__' : create_reprfn(cname,fields),
        })
        return C
    
    def create_sum(typ_name,t):
        T          = SC[typ_name]
        afields    = t.attributes
        for c in t.types:
            C      = create_sum_constructor(
                        typ_name, c.name, T,
                        c.fields + afields )
            assert (not hasattr(mod,c.name)), (
                f"name '{c.name}' conflict in module '{mod}'")
            setattr(T,c.name,C)
            setattr(mod,c.name,C)
        return T
    
    for nm,t in asdl_mod.types.items():
        if isinstance(t,asdl.Product):
            setattr(mod,nm,create_prod(nm,t))
        elif isinstance(t,asdl.Sum):
            setattr(mod,nm,create_sum(nm,t))
        else: assert false, "unexpected kind of asdl type"
            
    return mod

def ADT(asdl_str, ext_checks={}):
    """ Needs documentation?
    """
    asdl_ast = _asdl_parse(asdl_str)
    mod      = _build_classes(asdl_ast,ext_checks)
    # cache values in case we might want them
    mod._ext_checks = ext_checks
    mod._ast        = asdl_ast
    mod._defstr     = asdl_str
    return mod


_builtin_keymap = {
    'string'  : lambda x: x,
    'int'     : lambda x: x,
    'object'  : id,
    'float'   : lambda x: x,
    'bool'    : lambda x: x,
}

def _add_memoization(mod,whitelist,ext_key):
    asdl_mod = mod._ast
    
    keymap = _builtin_keymap.copy()
    for nm,fn in ext_key.items():
        keymap[nm] = fn
    for nm in asdl_mod.types:
        keymap[nm] = id
    
    def create_listkey(f):
        i = 'i' if f.name != i else 'ii'
        return (f"tuple([ K['{f.type}']({i}) "
                f"for {i} in {f.name} ]),")
    def create_optkey(f):
        return (f"None if {f.name} == None else "
                f"K['{f.type}']({f.name}),")
    
    def create_newfn(name, fields):
        if not name in whitelist: return
        T       = getattr(mod,name)
        
        argstr  = ', '.join([ f.name for f in fields ])
        keystr  = '('+(''.join([
            create_listkey(f) if f.seq else
            create_optkey(f)  if f.opt else
            f"K['{f.type}']({f.name}),"
            for f in fields
        ]))+')'
        
        exec_out = { 'T': T, 'K': keymap }
        exec_str = (f"def {name}_new(cls,{argstr}):\n"
                    f"    key = {keystr}\n"
                    f"    val = T._memo_cache.get(key)\n"
                    f"    if val == None:\n"
                    f"        val = super(T,cls).__new__(cls)\n"
                    f"        T._memo_cache[key] = val\n"
                    f"    return val")
        # un-comment this line to see what's
        # really going on
        #print(exec_str)
        exec(exec_str, exec_out)
        
        T._memo_cache = WeakValueDictionary({})
        T.__new__     = exec_out[name + '_new']
        
    def expand_sum(typ_name,t):
        T          = getattr(mod,typ_name)
        afields    = t.attributes
        for c in t.types:
            create_newfn(c.name, c.fields + afields)
    
    for nm,t in asdl_mod.types.items():
        if isinstance(t,asdl.Product):
            create_newfn(nm,t.fields)
        elif isinstance(t,asdl.Sum):
            expand_sum(nm,t)
        else: assert false, "unexpected kind of asdl type"

def memo(mod, whitelist, ext_key={}):
    _add_memoization(mod,whitelist,ext_key)




