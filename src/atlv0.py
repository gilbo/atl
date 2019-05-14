
import sys
from adt import ADT
from adt import memo as ADTmemo


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Tensor Language IR

IR = ADT("""
module IR {
    expr = Var      ( string name )
         | Const    ( float  val  )
         | Add      ( expr lhs, expr rhs )
         | Mul      ( expr lhs, expr rhs )
         | Pair     ( expr lhs, expr rhs )
         | Proj     ( int01 idx, expr arg )
         | Gen      ( string idxname, int range, expr body )
         | Sum      ( string idxname, int range, expr body )
         | Access   ( expr  base, index idx )
         -- implied multiplication of the bracket with body
         | Indicate ( pred  arg, expr body )
         -- important to express sharing of computation results
         | Let      ( string name, expr rhs, expr body )
    
    -- indices are drawn from a range s.t.
    -- 0 <= i < range
    
    pred    = Eq( index lhs, index rhs )
    
    type    = TNum    ()
            | TError  ()
            | TPair   (  type lhs, type rhs )
            | TTensor ( int range, type typ )
}
""", {
    'int01': lambda x: x == 0 or x == 1,
    'index': lambda x: (type(x) is int) or (type(x) is str),
})
ADTmemo(IR,['TNum','TError','TPair','TTensor'])
IR.tnum = IR.TNum()
IR.terr = IR.TError()

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Context object for making passes over the IR

class _Context:
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

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Type-Checking

class TCError(Exception):
    def __init(self, errs):
        errmsg = ("errors during typechecking:\n" +
                  ('\n'.join(errs)))
        super(TCError, self).__init__(errmsg)
        
class TypeChecker:
    def __init__(self, expr, initenv):
        self._ctxt   = _Context()
        for nm,typ in initenv.items():
            assert isinstance(typ, IR.type)
            self._ctxt.set(nm,typ)
        self._errors = []
        self._typ    = self.check(expr)
        self.report_errors()
    
    def _err(self, node, msg):
        # might want to discern location
        # via `node` eventually
        self._errors.append(msg)
    
    def report_errors(self):
        if len(self._errors) > 0:
            raise TCError('Found errors during typechecking:\n  '+
                          '\n  '.join(self._errors))
    
    def _get_ivar(self, node, name):
        idxrange = self._ctxt.get(name)
        if idxrange == None:
            self._err(node, f"index variable '{name}' was undefined")
        elif type(idxrange) is not int:
            self._err(node, f"variable '{name}' was "
                            f"not bound as an index variable")
        else: return idxrange
        # fail fall-through
        return None
    
    def _get_var(self, node, name):
        typ = self._ctxt.get(name)
        if typ == None: return None
        elif not isinstance(typ, IR.type):
            self._err(node, f"variable '{name}' was "
                            f"not bound as a normal variable")
            return T.terr
        # on success
        return typ
    
    def check(self, node):
        nclass = type(node)
        if   nclass is IR.Var:
            lookup = self._get_var(node, node.name)
            if lookup == None:
                self._err(node, f"expected variable '{node.name}' "
                                f"to have a type annotated.")
                return IR.terr
            return lookup

        elif nclass is IR.Const:
            return IR.tnum
        
        elif nclass is IR.Add or nclass is IR.Mul:
            ltyp = self.check(node.lhs)
            rtyp = self.check(node.rhs)
            typ  = IR.tnum if (ltyp == IR.tnum and
                               rtyp == IR.tnum) else IR.terr
            if ltyp != IR.tnum and ltyp != IR.terr:
                self._err(node,
                          f"expected number on left-hand-side "
                          f"of addition: {node}")
            if rtyp != IR.tnum and rtyp != IR.terr:
                self._err(node,
                          f"expected number on right-hand-side "
                          f"of addition: {node}")
            return typ
        
        elif nclass is IR.Pair:
            ltyp = self.check(node.lhs)
            rtyp = self.check(node.rhs)
            if ltyp == IR.terr or rtyp == IR.terr:
                return IR.terr
            else:
                return IR.TPair(ltyp,rtyp)
        
        elif nclass is IR.Proj:
            subtyp = self.check(node.arg)
            if subtyp == IR.terr: return IR.terr
            elif type(subtyp) is not T.TPair:
                self._err(node, f"Was expecting a pair as argument: {node}")
                return IR.terr
            elif node.idx == 0:   return subtyp.lhs
            else:                 return subtyp.rhs
        
        elif nclass is IR.Gen or nclass is IR.Sum:
            self._ctxt.push()
            self._ctxt.set(node.idxname, node.range)
            bodytyp = self.check(node.body)
            self._ctxt.pop()
            if   bodytyp == IR.terr: return IR.terr
            elif nclass is IR.Sum:
                if bodytyp == IR.tnum: return IR.tnum
                # otherwise
                self._err(node, f"Was expecting a number as argument: {node}")
                return IR.terr
            else: # nclass is IR.Gen
                return IR.TTensor(node.range, bodytyp)
        
        elif nclass is IR.Access:
            basetyp  = self.check(node.base)
            if basetyp == IR.terr: return IR.terr
            if not isinstance(basetyp,IR.TTensor):
                self._err(node, f"Was expecting a tensor to index: {node}")
                return IR.terr
            idxrange = self._get_ivar(node, node.idx)
            if idxrange == None: return IR.terr
            if idxrange != basetyp.range:
                self._err(node, f"index variable '{node.idx}' was bound "
                                f"to the range {idxrange}, but this tensor "
                                f"expects an index of range {basetyp.range}")
                return IR.terr
            # if reaching here, all checks passed
            # we can return the (accessed) de-tensor-ed type
            return basetyp.typ
        
        elif nclass is IR.Indicate:
            # need to check the predicate
            eqnode  = node.arg
            lrange  = self._get_ivar(node, eqnode.lhs)
            rrange  = self._get_ivar(node, eqnode.rhs)
            bodytyp = self.check(node.body)
            
            if lrange == None or rrange == None: pass
            elif lrange != rrange:
                self._err(node, f"index variables "
                                f"'{eqnode.lhs}' and '{eqnode.rhs}' "
                                f"in equality are drawn from different"
                                f"ranges: {lrange} and {rrange}")
            # can proceed to type-check regardless of errors
            # if we at least have the type of the body
            return bodytyp
        
        elif nclass is IR.Let:
            rtyp    = self.check(node.rhs)
            self._ctxt.push()
            self._ctxt.set(node.name, rtyp)
            bodytyp = self.check(node.body)
            self._ctxt.pop()
            return bodytyp
        
        else:
            assert false, "Unexpected expression class for {node}"

# use as
#  Typechecker(e,{'x':T.tnum})

def IR_check_python_value(typ, val):
    assert typ != IR.terr, "No values of type TErr"
    if typ == IR.tnum:
        if type(val) is not float:
            raise TypeError("Expected floating point value")
    elif isinstance(typ, IR.TPair):
        if type(val) is not tuple or len(val) != 2:
            raise TypeError("Expected pair value")
        L,R = val
        IR_check_python_value(typ.lhs, L)
        IR_check_python_value(typ.rhs, R)
    elif isinstance(typ, IR.TTensor):
        if type(val) is not list:
            raise TypeError("Expected list value")
        elif len(val) != typ.range:
            raise TypeError(f"Expected list of {typ.range} "
                            f"entries, but got {len(val)} entries")
        for i in range(typ.range):
            IR_check_python_value(typ.typ, val[i])
    else:
        assert False, f"{typ} should be a tensor-language type"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Operator Overloading for Convenience of IR construction

# e + e

def _expr_add_(lhs,rhs):
    if type(rhs) == int or type(rhs) == float:
        return IR.Add(lhs, IR.Const(float(rhs)))
    elif isinstance(rhs,IR.expr):
        return IR.Add(lhs,rhs)
    else:
        return NotImplemented
IR.expr.__add__ = _expr_add_
del _expr_add_

def _expr_radd_(rhs,lhs):
    if type(lhs) == int or type(lhs) == float:
        return IR.Add(IR.Const(float(lhs)), rhs)
    else:
        return NotImplemented
IR.expr.__radd__ = _expr_radd_
del _expr_radd_

# e * e

def _expr_mul_(lhs,rhs):
    if type(rhs) == int or type(rhs) == float:
        return IR.Mul(lhs, IR.Const(float(rhs)))
    elif isinstance(rhs,IR.expr):
        return IR.Mul(lhs,rhs)
    else:
        return NotImplemented
IR.expr.__mul__ = _expr_mul_
del _expr_mul_

def _expr_rmul_(rhs,lhs):
    if type(lhs) == int or type(lhs) == float:
        return IR.Mul(IR.Const(float(lhs)), rhs)
    else:
        return NotImplemented
IR.expr.__rmul__ = _expr_rmul_
del _expr_rmul_

# [p] * e

def _pred_mul_(lhs,rhs):
    if type(rhs) == int or type(rhs) == float:
        return IR.Indicate(lhs, IR.Const(float(rhs)))
    elif isinstance(rhs,IR.expr):
        return IR.Indicate(lhs,rhs)
    else:
        return NotImplemented
IR.pred.__mul__ = _pred_mul_
del _pred_mul_

def _pred_rmul_(rhs,lhs):
    if type(lhs) == int or type(lhs) == float:
        return IR.Indicate(rhs, IR.Const(float(lhs)))
    elif isinstance(lhs,IR.expr):
        return IR.Indicate(rhs,lhs)
    else:
        return NotImplemented
_pred_rmul_
del _pred_rmul_

# e[i]

def _expr_getitem_(expr,key):
    if not type(key) is tuple: key = (key,)
    # iterate over the key-tuple and glom on accesses
    for i in key:
        if type(i) == int or type(i) == str:
            expr = IR.Access(expr,i)
        else:
            raise TypeError(f"expected int or string in tensor index: {i}")
    return expr
IR.expr.__getitem__ = _expr_getitem_
del _expr_getitem_


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pretty Printing to Strings and to LaTeX (in Jupyter)

def IR_str_rep(e,prec=0):
    eclass = type(e)
    s      = "ERROR"
    if   eclass is IR.Var:
        s = e.name
    elif eclass is IR.Const:
        s = str(e.val)
    elif eclass is IR.Add:
        s = f"{IR_str_rep(e.lhs,2)} + {IR_str_rep(e.rhs,2)}"
        if prec > 2: s = f"({s})"
    elif eclass is IR.Mul:
        s = f"{IR_str_rep(e.lhs,3)} * {IR_str_rep(e.rhs,3)}"
        if prec > 3: s = f"({s})"
    elif eclass is IR.Pair:
        s = f"({IR_str_rep(e.lhs,0)},{IR_str_rep(e.rhs,0)})"
    elif eclass is IR.Proj:
        s = f"{IR_str_rep(e.arg,4)}.{e.idx}"
        if prec > 4: s = f"({s})"
    elif eclass is IR.Gen or eclass is IR.Sum:
        op = "+" if eclass is IR.Sum else "Gen"
        s = f"{op}({e.idxname}:{e.range}) {IR_str_rep(e.body,1)}"
        if prec > 1: s = f"({s})"
    elif eclass is IR.Access:
        s = f"{IR_str_rep(e.base,5)}[{e.idx}]"
        if prec > 5: s = f"({s})"
    elif eclass is IR.Indicate:
        assert isinstance(e.arg, IR.Eq), 'sanity: pred is Eq'
        s = f"[{e.arg.lhs}={e.arg.rhs}]*{IR_str_rep(e.body,3)}"
        if prec > 3: s = f"({s})"
    elif eclass is IR.Let:
        # note that this is ill-behaved formatting
        # for lets nested inside of expressions
        s = f"let {e.name} = {IR_str_rep(e.rhs,0)} in\n{IR_str_rep(e.body,0)}"
        if prec > 0: s = f"({s})"
    return s

IR.expr.__str__ = IR_str_rep


def IR_latex_str(e,prec=0):
    eclass = type(e)
    s      = "ERROR"
    if   eclass is IR.Var:
        s = e.name
    elif eclass is IR.Const:
        s = str(e.val)
    elif eclass is IR.Add:
        s = f"{IR_latex_str(e.lhs,2)} + {IR_latex_str(e.rhs,2)}"
        if prec > 2: s = f"\\left({s}\\right)"
    elif eclass is IR.Mul:
        s = f"{IR_latex_str(e.lhs,3)} \\cdot {IR_latex_str(e.rhs,3)}"
        if prec > 3: s = f"\\left({s}\\right)"
    elif eclass is IR.Pair:
        s = f"\\left({IR_latex_str(e.lhs,0)},{IR_latex_str(e.rhs,0)}\\right)"
    elif eclass is IR.Proj:
        s = f"\\pi_{{{e.idx}}} {IR_latex_str(e.arg,4)}"
        if prec > 4: s = f"\\left({s}\\right)"
    elif eclass is IR.Gen or eclass is IR.Sum:
        op = "\\sum" if eclass is IR.Sum else "\\boxplus"
        s = f"{op}_{{{e.idxname}:{e.range}}}\\ {IR_latex_str(e.body,1)}"
        if prec > 1: s = f"\\left({s}\\right)"
    elif eclass is IR.Access:
        s = f"{IR_latex_str(e.base,5)}[{e.idx}]"
        if prec > 5: s = f"\\left({s}\\right)"
    elif eclass is IR.Indicate:
        assert isinstance(e.arg, IR.Eq), 'sanity: pred is Eq'
        s = f"[{e.arg.lhs}={e.arg.rhs}]\\cdot {IR_latex_str(e.body,3)}"
        if prec > 3: s = f"\\left({s}\\right)"
    elif eclass is IR.Let:
        # note that this is ill-behaved formatting
        # for lets nested inside of expressions
        s = (f"\\begin{{array}}{{l}}"
             f" \\textrm{{let }} {e.name} = {IR_latex_str(e.rhs,0)}\\textrm{{ in}}\\\\"
             f" {IR_latex_str(e.body,0)}"
             f"\\end{{array}}")
        if prec > 0: s = f"\\left({s}\\right)"
    return s

def IR_latex_repr(e):
    return f"${IR_latex_str(e)}$"

IR.expr._repr_latex_ = IR_latex_repr


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python Execution for Reference

class Interpreter():
    def __init__(self,env):
        self._init_vals = {}
        for name in env:
            val = env[name]
            self._init_vals[name] = val
    
    def run(self,e):
        self._vals = _Context()
        for nm,v in self._init_vals.items():
            self._vals.set(nm,v)
        return self._exec(e)
    
    def _get_val(self,nm):
        v = self._vals.get(nm)
        if v is None: raise KeyError(f"Did not find variable '{nm}'")
        return v
    
    def _get_ival(self,idx):
        if type(idx) is int: return idx
        else: return self._get_val(idx)
    
    def _exec(self,e):
        eclass = type(e)
        if   eclass is IR.Var:
            return self._get_val(e.name)
        
        elif eclass is IR.Const:
            return e.val
        
        elif eclass is IR.Add:
            return self._exec(e.lhs) + self._exec(e.rhs)
        
        elif eclass is IR.Mul:
            return self._exec(e.lhs) * self._exec(e.rhs)
        
        elif eclass is IR.Pair:
            return ( self._exec(e.lhs), self._exec(e.rhs) )
        
        elif eclass is IR.Proj:
            return self._exec(e.arg)[e.idx]
        
        elif eclass is IR.Gen:
            xs = []
            self._vals.push()
            for i in range(e.range):
                self._vals.set(e.idxname,i)
                xs.append( self._exec(e.body) )
            self._vals.pop()
            return xs
        
        elif eclass is IR.Sum:
            acc = 0.0
            self._vals.push()
            for i in range(e.range):
                self._vals.set(e.idxname,i)
                acc += self._exec(e.body)
            self._vals.pop()
            return acc
        
        elif eclass is IR.Access:
            base = self._exec(e.base)
            i = self._get_ival(e.idx)
            return base[i]
        
        elif eclass is IR.Indicate:
            assert type(e.arg) is IR.Eq, 'expect only Eq pred'
            i = self._get_ival(e.arg.lhs)
            j = self._get_ival(e.arg.rhs)
            if i == j:
                return self._exec(e.body)
            else:
                return 0.0
        
        elif eclass is IR.Let:
            rval = self._exec(e.rhs)
            self._vals.push()
            self._vals.set(e.name,rval)
            body = self._exec(e.body)
            self._vals.pop()
            return body
        
        else:
            assert False, "Unexpected Exec Case"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Functions

class Func():
    def __init__(self,argtyps,expr):
        self._expr    = expr
        # construct the args two ways
        self._arglist = []
        self._argdict = {}
        assert type(argtyps) is list
        for nm,typ in argtyps:
            assert type(nm) is str, 'expected argument names as strings'
            assert isinstance(typ,IR.type), 'expected type for argument'
            self._arglist.append( (nm,typ) )
            self._argdict[nm] = typ
        # run basic type-checking
        TypeChecker(expr,self._argdict).report_errors()

    def interpret(self, *args, **kwargs):
        call_args   = {}

        # check that the right number of arguments were used
        n_call      = len(args) + len(kwargs)
        n_args      = len(self._arglist)
        if n_call != n_args:
            raise TypeError(f"expected {n_args} arguments, "
                            f"but was called with {n_call}")

        # fill out call_args with supplied named arguments
        for nm in kwargs:
            typ     = self._argdict.get(nm)
            if typ is None:
                raise TypeError(f"argument '{nm}' is not an argument of "
                                f"this tensor function")
            else:
                IR_check_python_value(typ,kwargs[nm])
                call_args[nm] = kwargs[nm]

        # then fill in the remainder with the unnamed arguments
        arg_i   = 0
        for nm,typ in self._arglist:
            if not nm in call_args:
                assert(arg_i < len(args))
                val     = args[arg_i]
                arg_i   = arg_i + 1
                IR_check_python_value(typ,val)
                call_args[nm] = val

        # finally, execute
        return Interpreter(call_args).run(self._expr)

    def __call__(self, *args, **kwargs):
        return self.interpret(*args,**kwargs)


if __name__ == '__main__':
    xv  = [4.,7.,1.]
    Av  = [[5.,2.,0.],[2.2,0.,4.5],[0.,6.1,3.3]]
    cv  = [0.,0.,1.]
    n   = 3
    R   = IR.tnum
    Rn  = IR.TTensor(n,R)
    Rnn = IR.TTensor(n,Rn)
    x   = IR.Var('x')
    A   = IR.Var('A')
    c   = IR.Var('c')
    i   = 'i'
    j   = 'j'

    y       = IR.Gen(i,n, c[i] + IR.Sum(j,n, A[i,j] * x[j] ))
    tr      = IR.Sum(i,n, A[i,i])
    diag    = IR.Gen(i,n,IR.Gen(j,n, IR.Eq(i,j) * x[i] ))

    Fy      = Func([('A',Rnn),('x',Rn),('c',Rn)], y)
    Ftr     = Func([('A',Rnn)], tr)
    Fdiag   = Func([('x',Rn)],  diag)

    print( Fy(A = Av, x = xv, c = cv) )
    print( Ftr(Av) )
    print( Fdiag(xv) )

