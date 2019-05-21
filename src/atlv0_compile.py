import sys
from adt import ADT
from adt import memo as ADTmemo
from atlv0 import IR as IRv0
from atlv0 import Func
from atlv0 import TCError
from atlv0 import _Context
import numpy as np
import time
from halide import *


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Symbol objects so we don't have to use strings for names

class Sym:
    unq_count   = 1
    
    def __init__(self,nm):
        self._nm    = nm
        self._id    = Sym.unq_count
        Sym.unq_count += 1
    
    def __str__(self):
        return self._nm
    
    def __repr__(self):
        return f"{self._nm}${self._id}"
    
    def name(self):
        return self._nm
    
    def copy(self):
        return Sym(self._nm)
    


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Typed Version of the IR; also uses symbols now

IR = ADT("""
module IR {
    expr = Var      ( symbol name )
         | Const    ( float  val  )
         | Add      ( expr lhs, expr rhs )
         | Mul      ( expr lhs, expr rhs )
         | Pair     ( expr lhs, expr rhs )
         | Proj     ( int01 idx, expr arg )
         | Gen      ( symbol idxname, int range, expr body )
         | Sum      ( symbol idxname, int range, expr body )
         | Access   ( expr  base, index idx )
         -- implied multiplication of the bracket with body
         | Indicate ( pred  arg, expr body )
         -- important to express sharing of computation results
         | Let      ( symbol name, expr rhs, expr body )
         attributes( type typ )
    
    -- indices are drawn from a range s.t.
    -- 0 <= i < range
    
    pred    = Eq( index lhs, index rhs )
}
""", {
    'int01':  lambda x:  x == 0 or x == 1,
    'index':  lambda x:  (type(x) is int) or (type(x) is Sym),
    'symbol': lambda x: type(x) is Sym,
    'type':   lambda x:  isinstance(x, IRv0.type)
})
# copy over types from the old module...
for nm in ['tnum','terr','type','TNum','TError','TPair','TTensor']:
    setattr(IR,nm,getattr(IRv0,nm))
del nm

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _IR_typ_str_rep(t):
    tclass = type(t)
    s      = "ERROR"
    if tclass is IR.TNum:
        s  = "R"
    elif tclass is IR.TPair:
        s  = f"({_IR_typ_str_rep(t.lhs)},{_IR_typ_str_rep(t.rhs)})"
    elif tclass is IR.TTensor:
        s  = f"[{t.range}]{_IR_typ_str_rep(t.typ)}"
    return s
IR.type.__str__ = _IR_typ_str_rep
    
def _IR_str_rep(e,prec=0,indent=""):
    def sub(e,p):
        return _IR_str_rep(e,p,indent)
    eclass = type(e)
    s      = "ERROR"
    if   eclass is IR.Var:
        s = e.name
    elif eclass is IR.Const:
        s = str(e.val)
    elif eclass is IR.Add:
        s = f"{sub(e.lhs,2)} + {sub(e.rhs,2)}"
        if prec > 2: s = f"({s})"
    elif eclass is IR.Mul:
        s = f"{sub(e.lhs,3)} * {sub(e.rhs,3)}"
        if prec > 3: s = f"({s})"
    elif eclass is IR.Pair:
        s = f"({sub(e.lhs,0)},{sub(e.rhs,0)})"
    elif eclass is IR.Proj:
        s = f"{sub(e.arg,4)}.{e.idx}"
        if prec > 4: s = f"({s})"
    elif eclass is IR.Gen or eclass is IR.Sum:
        op = "+" if eclass is IR.Sum else "Gen"
        s = f"{op}({e.idxname}:{e.range}) {sub(e.body,1)}"
        if prec > 1: s = f"({s})"
    elif eclass is IR.Access:
        s = f"{sub(e.base,5)}[{e.idx}]"
        if prec > 5: s = f"({s})"
    elif eclass is IR.Indicate:
        assert isinstance(e.arg, IR.Eq), 'sanity: pred is Eq'
        s = f"[{e.arg.lhs}={e.arg.rhs}]*{sub(e.body,3)}"
        if prec > 3: s = f"({s})"
    elif eclass is IR.Let:
        # note that this is ill-behaved formatting
        # for lets nested inside of expressions
        rhs  = sub(e.rhs,0)
        body = sub(e.body,0)
        s = f"let {e.name} : {e.rhs.typ} = {rhs} in\n{indent}{body}"
        if prec > 0: s = f"({s})"
    return s

IR.expr.__str__ = _IR_str_rep

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class TypeChecker:
    def __init__(self, expr, initsymtyps):
        self._ctxt   = _Context()
        for nm,symtyp in initsymtyps.items():
            assert type(symtyp[0]) is Sym
            assert isinstance(symtyp[1], IR.type)
            self._ctxt.set(nm,symtyp)
        self._errors = []
        self._out_ir = self.check(expr)
        self.report_errors()

    def get_out_ir(self):
        return self._out_ir
    
    def _err(self, node, msg):
        # might want to discern location
        # via `node` eventually
        self._errors.append(msg)
    
    def report_errors(self):
        if len(self._errors) > 0:
            raise TCError('Found errors during typechecking:\n  '+
                          '\n  '.join(self._errors))
    
    def _get_ivar(self, node, name):
        symtyp  = self._ctxt.get(name)
        if symtyp == None:
            self._err(node, f"index variable '{name}' was undefined")
        elif type(symtyp[1]) is not int:
            self._err(node, f"variable '{name}' was "
                            f"not bound as an index variable")
        else: return symtyp
        # on failure fallthrough
        return (Sym(name),None)
    
    def _get_var(self, node, name):
        symtyp  = self._ctxt.get(name)
        if symtyp == None: 
            self._err(node, f"variable '{name}' was undefined")
        elif not isinstance(symtyp[1], IR.type):
            self._err(node, f"variable '{name}' was "
                            f"not bound as a variable")
        else: return symtyp
        # on failure fallthrough
        return (Sym(name),IR.terr)
    
    def check(self, node):
        nclass = type(node)
        if   nclass is IRv0.Var:
            sym,typ = self._get_var(node, node.name)
            if typ == None: typ = IR.terr
            return IR.Var(sym, typ)

        elif nclass is IRv0.Const:
            return IR.Const(node.val,IR.tnum)
        
        elif nclass is IRv0.Add or nclass is IRv0.Mul:
            lhs = self.check(node.lhs)
            rhs = self.check(node.rhs)
            typ = IR.tnum if (lhs.typ == IR.tnum and
                              rhs.typ == IR.tnum) else IR.terr
            if lhs.typ != IR.tnum and lhs.typ != IR.terr:
                self._err(node,
                          f"expected number on left-hand-side "
                          f"of addition: {node}")
            if rhs.typ != IR.tnum and rhs.typ != IR.terr:
                self._err(node,
                          f"expected number on right-hand-side "
                          f"of addition: {node}")
            if nclass is IRv0.Add:
                return IR.Add(lhs,rhs,typ)
            else:
                return IR.Mul(lhs,rhs,typ)
        
        elif nclass is IRv0.Pair:
            lhs = self.check(node.lhs)
            rhs = self.check(node.rhs)
            typ = IR.terr
            if lhs.typ != IR.terr and rhs.typ != IR.terr:
                typ = IR.TPair(lhs.typ,rhs.typ)
            return IR.Pair(lhs,rhs,typ)
        
        elif nclass is IRv0.Proj:
            arg = self.check(node.arg)
            typ = IR.terr
            if   arg.typ == IR.terr: pass
            elif type(arg.typ) is not IR.TPair:
                self._err(node, f"Was expecting a pair as argument: {node}")
            elif node.idx == 0: typ = arg.typ.lhs
            else:               typ = arg.typ.rhs
            return IR.Proj(node.idx,arg,typ)
        
        elif nclass is IRv0.Gen or nclass is IRv0.Sum:
            self._ctxt.push()
            newsym  = Sym(node.idxname)
            self._ctxt.set(node.idxname, (newsym,node.range) )
            body    = self.check(node.body)
            self._ctxt.pop()
            if   nclass is IRv0.Sum:
                typ = body.typ
                if typ != IR.tnum and typ != IR.terr:
                    self._err(node, f"Was expecting a number as body: {node}")
                    typ = IR.terr
                return IR.Sum(newsym,node.range,body,typ)
            else: # nclass is IRv0.Gen
                typ = IR.terr
                if body.typ != IR.terr:
                    typ = IR.TTensor(node.range, body.typ)
                return IR.Gen(newsym,node.range,body,typ)
        
        elif nclass is IRv0.Access:
            base    = self.check(node.base)
            sym,rng = self._get_ivar(node, node.idx)
            typ     = IR.terr
            if base.typ == IR.terr: pass
            elif not isinstance(base.typ,IR.TTensor):
                self._err(node, f"Was expecting a tensor to index: {node}")
            elif rng == None:  pass
            elif rng != base.typ.range:
                self._err(node, f"index variable '{node.idx}' was bound "
                                f"to the range {rng}, but this tensor "
                                f"expects an index of range {base.typ.range}")
            else:
                typ = base.typ.typ
            return IR.Access(base,sym,typ)
        
        elif nclass is IRv0.Indicate:
            # need to check the predicate
            eqnode      = node.arg
            lsym, lrng  = self._get_ivar(node, eqnode.lhs)
            rsym, rrng  = self._get_ivar(node, eqnode.rhs)
            body        = self.check(node.body)
            
            if   lrng == None or rrng == None: pass
            elif lrng != rrng:
                self._err(node, f"index variables "
                                f"'{eqnode.lhs}' and '{eqnode.rhs}' "
                                f"in equality are drawn from different"
                                f"ranges: {lrng} and {rrng}")
            # can proceed with the body type regardless of errors
            return IR.Indicate( IR.Eq(lsym,rsym), body, body.typ)
        
        elif nclass is IRv0.Let:
            rhs     = self.check(node.rhs)
            self._ctxt.push()
            newsym  = Sym(node.name)
            self._ctxt.set(node.name, (newsym,rhs.typ) )
            body    = self.check(node.body)
            self._ctxt.pop()
            return IR.Let(newsym,rhs,body,body.typ)
        
        else:
            assert false, "Unexpected expression class for {node}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def _typ_has_pairs(t):
    tclass = type(t)
    if tclass is IR.TPair: return True
    elif tclass is IR.TTensor:
        return _typ_has_pairs(t.typ)
    else: # tnum or terr
        return False

def _typ_SoA_transform(t,rngs=[]):
    tclass = type(t)
    if t is IR.tnum or t is IR.terr:
        # possibly unroll the ranges stack here
        if len(rngs) > 0:
            for r in reversed(rngs):
                t = IR.TTensor(r,t)
        return t
    elif tclass is IR.TTensor:
        rngs = rngs.copy()
        rngs.append(t.range)
        return _typ_SoA_transform(t.typ,rngs)
    elif tclass is IR.TPair:
        return IR.TPair( _typ_SoA_transform(t.lhs,rngs),
                         _typ_SoA_transform(t.rhs,rngs) )
    else: assert false, "impossible case"

class PairEliminator:
    def __init__(self,expr,arg_symtyps,rettyp):
        self._expr      = expr
        self._ctxt      = _Context()
        
        if _typ_has_pairs(rettyp):
            raise TypeError(f"Cannot eliminate pairs from a function "
                            f"whose return type is a pair")
        
        for sym,typ in arg_symtyps.values():
            if _typ_has_pairs(typ):
                raise TypeError(f"Cannot eliminate pairs from a function "
                                f"with pair-typed arguments")
            self._ctxt.set(sym,IR.Var(sym,typ))
        
        # do elimination
        self._out_expr  = self.pushdown(expr)
    
    def get_result(self):
        return self._out_expr
    
    def _get_ivar_sub(self,nm):
        i = self._ctxt.get(nm)
        assert not i is None, f'should be caught earlier: {nm}'
        return i
    
    def _get_var_sub(self,nm):
        e = self._ctxt.get(nm)
        assert not e is None, f'should be caught earlier: {nm}'
        return e
    
    def pushdown(self, e, projstk = []):
        eclass = type(e)
        if   eclass is IR.Var:
            subst_e = self._get_var_sub(e.name)
            # prevent infinite recursion and needless duplications
            if (type(subst_e) is IR.Var and
                subst_e.name == e.name and
                subst_e.typ  == e.typ):
                    assert(len(projstk) == 0)
                    return e
            else: # need to continue pushdown
                return self.pushdown(subst_e, projstk)
        
        elif eclass is IR.Const:
            assert len(projstk) == 0
            return e
        
        elif eclass is IR.Add or eclass is IR.Mul:
            assert len(projstk) == 0
            lhs = self.pushdown(e.lhs, projstk)
            rhs = self.pushdown(e.rhs, projstk)
            return eclass(lhs, rhs, e.typ)
        
        # deconstruct it!
        elif eclass is IR.Pair:
            assert len(projstk) > 0
            proj_i = projstk.pop()
            if proj_i == 0:
                return self.pushdown(e.lhs, projstk)
            else:
                return self.pushdown(e.rhs, projstk)
        
        elif eclass is IR.Proj:
            projstk.append(e.idx)
            return self.pushdown(e.arg, projstk)
        
        elif eclass is IR.Gen or eclass is IR.Sum:
            self._ctxt.push()
            idxname = e.idxname.copy()
            self._ctxt.set(e.idxname,idxname)
            body    = self.pushdown(e.body, projstk)
            if eclass is IR.Gen:
                TensorType = IR.TTensor(e.range, body.typ)
                return IR.Gen(idxname, e.range, body, TensorType)
            else:
                assert body.typ == IR.tnum
                return IR.Sum(idxname, e.range, body, body.typ)
            
        elif eclass is IR.Access:
            base    = self.pushdown(e.base, projstk)
            idx     = self._get_ivar_sub(e.idx)
            assert type(base.typ) is IR.TTensor
            return IR.Access(base, idx, base.typ.typ)
            
        elif eclass is IR.Indicate:
            lidx    = self._get_ivar_sub(e.arg.lhs)
            ridx    = self._get_ivar_sub(e.arg.rhs)
            body    = self.pushdown(e.body, projstk)
            return IR.Indicate(IR.Eq(lidx,ridx), body, body.typ)
            
        elif eclass is IR.Let:
            soa_typ = _typ_SoA_transform(e.rhs.typ)

            # unpack the soa_typ into projections
            binds   = []
            def soa_unpack(nm,T,projstk=[]):
                if type(T) is IR.TPair:
                    projstk.insert(0,0)
                    lhs     = soa_unpack(nm+'0',T.lhs,projstk)
                    projstk.pop()
                    projstk.insert(0,1)
                    rhs     = soa_unpack(nm+'1',T.rhs,projstk)
                    projstk.pop()
                    assert lhs.typ == T.lhs
                    assert rhs.typ == T.rhs
                    return IR.Pair(lhs,rhs,T)
                else:
                    rval    = self.pushdown(e.rhs,projstk.copy())
                    sym     = Sym(nm)
                    assert rval.typ == T
                    binds.append((sym,rval))
                    return IR.Var(sym,T)
            nm = e.name.name() + ('_' if (type(soa_typ) is IR.TPair) else '')
            subst = soa_unpack(nm, soa_typ)
            
            # bind the soa-transformed variables, and rewrite body
            self._ctxt.push()
            self._ctxt.set(e.name,subst)
            # ensure termination at the new let-bound variables
            for sym,rhs in binds:
                self._ctxt.set(sym,IR.Var(sym,rhs.typ))
            body    = self.pushdown(e.body,projstk)
            self._ctxt.pop()
            
            # construct the resulting let-binding chain
            for sym,rhs in reversed(binds):
                body = IR.Let(sym,rhs,body,body.typ)
            return body


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class LetFlatten:
    def __init__(self,expr,arg_symtyps,rettyp):
        self._expr      = expr
        #self._ctxt      = _Context()
        
        # lift
        binds, ret_e    = self.letlift(expr)
        
        # construct the final let-chained expression
        e   = ret_e
        for nm,rhs in reversed(binds):
            e = IR.Let(nm,rhs,e,e.typ)
        self._out_expr  = e
    
    def get_result(self):
        return self._out_expr
    
    def letlift(self, e):
        eclass = type(e)
        assert not eclass is IR.Pair, "pairs should be eliminated"
        assert not eclass is IR.Proj, "pairs should be eliminated"
        if   eclass is IR.Var:
            return [],e
        elif eclass is IR.Const:
            return [],e
        
        elif eclass is IR.Add or eclass is IR.Mul:
            lbind, lhs  = self.letlift(e.lhs)
            rbind, rhs  = self.letlift(e.rhs)
            
            return lbind + rbind, eclass(lhs, rhs, e.typ)
        
        elif eclass is IR.Gen or eclass is IR.Sum:
            binds, body = self.letlift(e.body)
            ctxt        = _Context()
            i           = e.idxname
            rng         = e.range
            
            new_binds   = []
            for nm,rhs in binds:
                rhs     = self.subst(ctxt,rhs)
                T       = rhs.typ
                TensorT = IR.TTensor(rng,T)
                new_rhs = IR.Gen(i,rng,rhs,TensorT)
                ctxt.set( nm, IR.Access(IR.Var(nm,TensorT), i, T) )
                new_binds.append( (nm, new_rhs) )
            new_body    = eclass(i,rng, self.subst(ctxt,body), e.typ)
            
            return new_binds, new_body
            
        elif eclass is IR.Access:
            binds, base = self.letlift(e.base)
            return binds, IR.Access(base, e.idx, e.typ)
            
        elif eclass is IR.Indicate:
            binds, body = self.letlift(e.body)
            return binds, IR.Indicate(e.arg, body, e.typ)
            
        elif eclass is IR.Let:
            binds0, rhs     = self.letlift(e.rhs)
            binds1, body    = self.letlift(e.body)
            binds           = binds0 + [(e.name,rhs)] + binds1
            return binds, body
    
    def subst(self, env, e):
        eclass = type(e)
        assert not eclass is IR.Pair, "pairs should be eliminated"
        assert not eclass is IR.Proj, "pairs should be eliminated"
        assert not eclass is IR.Let
        if   eclass is IR.Var:
            sub = env.get(e.name)
            return e if sub is None else sub
        
        elif eclass is IR.Const:
            return e
        
        elif eclass is IR.Add or eclass is IR.Mul:
            lhs = self.subst(env, e.lhs)
            rhs = self.subst(env, e.rhs)
            return eclass(lhs,rhs,e.typ)
        
        elif eclass is IR.Gen or eclass is IR.Sum:
            body    = self.subst(env, e.body)
            return eclass(e.idxname, e.range, body, e.typ)
            
        elif eclass is IR.Access:
            base    = self.subst(env, e.base)
            return IR.Access(base, e.idx, e.typ)
            
        elif eclass is IR.Indicate:
            body    = self.subst(env, e.body)
            return IR.Indicate(e.arg, body, e.typ)



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class GenNormalize:
    def __init__(self,expr,arg_symtyps,rettyp):
        self._expr = expr
        
        # break out into a list of bindings and return expr
        binds, ret = self.to_block(expr)
        
        # normalize each expr
        rebinds    = []
        for sym,rhs in binds:
            r      = self.final_lift_gen(rhs)
            rebinds.append((sym,r))
        ret        = self.final_lift_gen(ret)
        
        self._out_expr  = self.from_block(rebinds,ret)
    
    def get_result(self):
        return self._out_expr
    
    def to_block(self, e):
        binds = []
        while type(e) is IR.Let:
            binds.append((e.name,e.rhs))
            e = e.body
        return binds, e
    
    def from_block(self, binds, ret):
        e = ret
        for sym,rhs in reversed(binds):
            e = IR.Let(sym,rhs,e,e.typ)
        return e
    
    def wrap_ind(self, inds, e):
        for p in inds:
            e = IR.Indicate(p, e, e.typ)
        return e
    
    def final_lift_gen(self, e):
        gen, ind, e = self.lift_gen(e)
        e           = self.wrap_ind(ind,e)
        for idx,rng in gen:
            T       = IR.TTensor(rng,e.typ)
            e       = IR.Gen(idx,rng,e,T)
        return e
    
    def lift_gen(self, e):
        eclass = type(e)
        assert not eclass is IR.Pair, "pairs should be eliminated"
        assert not eclass is IR.Proj, "pairs should be eliminated"
        assert not eclass is IR.Let
        if   eclass is IR.Var or eclass is IR.Const:
            return [], [], e
        
        elif eclass is IR.Add or eclass is IR.Mul:
            lgen, lind, lhs = self.lift_gen(e.lhs)
            rgen, rind, rhs = self.lift_gen(e.rhs)
            assert len(lgen) == 0
            assert len(rgen) == 0
            if eclass is IR.Add:
                lhs = self.wrap_ind(lind,lhs)
                rhs = self.wrap_ind(rind,rhs)
                return [], [], IR.Add(lhs,rhs,e.typ)
            else:
                return [], lind + rind, IR.Mul(lhs,rhs,e.typ)
        
        elif eclass is IR.Gen:
            gen, ind, body  = self.lift_gen(e.body)
            gen.append((e.idxname, e.range))
            return gen, ind, body
        
        elif eclass is IR.Sum:
            gen, ind, body  = self.lift_gen(e.body)
            assert len(gen) == 0
            # go through indicators and maybe find sum collapse
            i = e.idxname
            j = None
            for p in ind:
                if   p.lhs == i: j = p.rhs; break
                elif p.rhs == i: j = p.lhs; break
            if j is None:
                body = IR.Sum(i,e.range,body,body.typ)
            else:
                ind, body   = self.subst(i,j,ind,body)
            return [], ind, body
            
        elif eclass is IR.Access:
            gen, ind, base  = self.lift_gen(e.base)
            if len(gen) > 0:
                i,r         = gen.pop()
                ind, base   = self.subst(i,e.idx,ind,base)
                return gen, ind, base
            else:
                return gen, ind, IR.Access(base, e.idx, e.typ)
            
        elif eclass is IR.Indicate:
            gen, ind, body  = self.lift_gen(e.body)
            # sanity!
            for i,r in gen:
                assert i != e.arg.lhs and i != e.arg.rhs
            ind.append(e.arg)
            return gen, ind, body
            
    def sub_ind(self, old, new, p):
        assert type(p) is IR.Eq
        lhs = new if p.lhs == old else p.lhs
        rhs = new if p.rhs == old else p.rhs
        return None if lhs == rhs else IR.Eq(lhs,rhs)
    
    def subst(self, old, new, ind, e):
        # if there are indicators, process separately
        if len(ind) > 0:
            new_ind = []
            for p in ind:
                p = self.sub_ind(old,new,p)
                if not p is None: new_ind.append(p)
            # substitute the expression without indicators
            _, e = self.subst(old,new,[],e)
            return new_ind, e
        
        # the usual case for things other than indicator lists
        eclass = type(e)
        assert not eclass is IR.Pair, "pairs should be eliminated"
        assert not eclass is IR.Proj, "pairs should be eliminated"
        assert not eclass is IR.Let
        assert not eclass is IR.Gen
        if   eclass is IR.Var or eclass is IR.Const:
            return [], e
        
        elif eclass is IR.Add or eclass is IR.Mul:
            _, lhs  = self.subst(old,new,ind,e.lhs)
            _, rhs  = self.subst(old,new,ind,e.rhs)
            return [], eclass(lhs,rhs,e.typ)
        
        elif eclass is IR.Sum:
            # probably shouldn't happen?
            if e.idxname == old:
                return e
            _, body = self.subst(old,new,[],e.body)
            return [], eclass(e.idxname, e.range, body, e.typ)
            
        elif eclass is IR.Access:
            idx     = new if e.idx == old else e.idx
            _, base = self.subst(old,new,[],e.base)
            return [], IR.Access(base, idx, e.typ)
            
        elif eclass is IR.Indicate:
            arg     = self.sub_ind(old,new,e.arg)
            _, body = self.subst(old,new,[],e.body)
            if arg is None: return [], body
            else:           return [], IR.Indicate(arg,body,body.typ)




# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def _get_block_norm(e):
    stmts   = []
    expr    = e
    while type(expr) is IR.Let:
        nm      = expr.name
        e       = expr.rhs
        gens    = []
        while type(e) is IR.Gen:
            i   = e.idxname
            r   = e.range
            gens.append((i,r))
            e   = e.body
        stmts.append((nm,gens,e))
        expr    = expr.body
    body_gens   = []
    while type(expr) is IR.Gen:
        i       = expr.idxname
        r       = expr.range
        body_gens.append((i,r))
        expr    = expr.body
        
        
    # stmts has form
    #   [( var_name, [( i_name, range )], (body_gens, body_expr) )]
    return stmts, (body_gens,expr)


def _type_value_match(typ,val):
    if type(typ) is IR.Pair:
        raise TypeError("Pairs unsupported")
    elif type(typ) is IR.TTensor:
        # unroll tensor
        shape = []
        while type(typ) is IR.TTensor:
            shape.append(typ.range)
            typ = typ.typ
        if type(typ) is IR.Pair:
            raise TypeError("Pairs unsupported")
        assert typ is IR.tnum
        
        if type(val) != np.ndarray:
            raise TypeError("Expected 'numpy.ndarray' type value")
        # check shape
        if len(shape) != len(val.shape):
            raise TypeError(f"Expected {len(shape)} dims, but got "
                            f"an ndarray with {len(val.shape)}")
        for i,d in enumerate(shape):
            if d != val.shape[i]:
                raise TypeError(f"expected dimension {i} of tensor shape "
                                f"to be {d}, but it was {val.shape[i]}")
        
    elif typ is IR.tnum:
        if type(val) != float and type(val) != int:
            raise TypeError("Expected 'float' or 'int' type value")


def _ndarray_to_halide_buf(a):
    def typ_convert(dt):
        t = dt.type
        if t is float:
          assert sys.float_info.max_exp == 1024
        if t is np.float64 or t is float:
          return halide_type_t(C.type_float,64,1)
        else: raise TypeError("expect only float64 numeric data")
    
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


class Halide_CodeGen:
    def __init__(self,expr,arg_symtyps,rettyp):        
        # break out into a list of bindings and return expr
        stmts, ret  = _get_block_norm(expr)
        self._orig_stmts = stmts
        self._orig_ret  = ret
        
        self._ctxt      = _Context()
        self._first_run = True
        
        # helper function to process tensor types
        def shape_dim(T):
            if not type(T) is IR.TTensor: return []
            else: return [T.range] + shape_dim(T.typ)
        
        # create Halide inputs for each argument
        # inputs may be 'param's (scalars) or
        #               'img_param's (tensors)
        Hf64        = halide_type_t(C.type_float,64,1)
        arg_params  = {}
        for nm,typ in arg_symtyps.values():
            nm_bytes    = repr(nm).encode('utf-8')
            shape       = shape_dim(typ)
            if len(shape) == 0: # scalar case
                P = C.hwrap_new_param(nm_bytes,Hf64)
                arg_params[nm.name()] = P
                # also bind the arg symbol to an Expr
                E = C.hwrap_param_to_expr(P)
                self._ctxt.set( nm, E )
            else: # tensor case
                Img = C.hwrap_new_img(nm_bytes,len(shape),Hf64)
                arg_params[nm.name()] = Img
                # also bind the arg symbol to a Func
                F   = C.hwrap_img_to_func(Img)
                self._ctxt.set( nm, F )
                # Halide auto-scheduling requires an estimate
                # of tensor size, which we extract from the type
                for i,r in enumerate(shape):
                    C.hwrap_set_img_bound_estimate(Img,i,
                        C.hwrap_i32_to_expr(0),C.hwrap_i32_to_expr(r))
        self._arg_params    = arg_params
        self._arg_typs      = { nm : symtyp[1]
                                for nm,symtyp in arg_symtyps.items() }
        
        # compile each statement
        for name,gens,body in stmts:
            self._compile_stmt(name,gens,body)
        
        # compile the return expression as a statement
        self._ret_sym       = Sym('return')
        self._ret_Func      = self._compile_stmt(self._ret_sym,
                                                 ret[0], ret[1])
        self._ret_typ       = rettyp
        # also provide output estimates for auto-scheduling
        if type(rettyp) is IR.TTensor:
            for i,r in enumerate(shape_dim(rettyp)):
                C.hwrap_set_func_bound_estimate(self._ret_Func,i,
                    C.hwrap_i32_to_expr(0),C.hwrap_i32_to_expr(r))
        else: # scalar temporaries encoded as length 1 arrays
            C.hwrap_set_func_bound_estimate(self._ret_Func,0,
                C.hwrap_i32_to_expr(0),C.hwrap_i32_to_expr(1))
        
    
    def _compile_stmt(self, name, gens, expr):
        name_bytes  = repr(name).encode('utf-8')
        F           = C.hwrap_new_func(name_bytes)
        
        self._ctxt.push()
        
        # create index variables
        n_dims      = len(gens)
        i_var_arr   = None
        handles     = [] # to prevent garbage collection
        if n_dims == 0:
            i       = Sym(name.name()+"_0idx")
            i_bytes = repr(i).encode('utf-8')
            V       = C.hwrap_new_var(i_bytes)
            handles.append(V)
            i_var_arr   = (hw_var_t * 1)(V)
            n_dims      = 1
        else:
            i_var_arr   = (hw_var_t * n_dims)()
            for k,ir in enumerate(gens):
                i, r    = ir
                i_bytes = repr(i).encode('utf-8')
                V       = C.hwrap_new_var(i_bytes)
                E       = C.hwrap_var_to_expr(V)
                handles.append(V)
                handles.append(E)
                # pack var into lhs array
                i_var_arr[k] = V
                # store expr in context
                self._ctxt.set(i,E)
        
        # compile rhs expr
        rhs = self._compile_expr(expr)
        self._ctxt.pop()
        
        # add the statement to the program
        C.hwrap_pure_def(F,n_dims,i_var_arr,rhs)
        
        # add the function to the context
        if len(gens) == 0:
            zero    = C.hwrap_i32_to_expr(0)
            z_arr   = (hw_expr_t * 1)(zero)
            E       = C.hwrap_access_func(F,1,z_arr)
            self._ctxt.set(name,E)
        else:
            self._ctxt.set(name,F)
        return F
    
    def _compile_expr(self,e):
        eclass = type(e)
        assert not eclass is IR.Pair, "pairs should be eliminated"
        assert not eclass is IR.Proj, "pairs should be eliminated"
        assert not eclass is IR.Let
        assert not eclass is IR.Gen
        if   eclass is IR.Var:
            expr = self._ctxt.get(e.name)
            assert expr != None
            return expr
        
        elif eclass is IR.Const:
            expr = C.hwrap_f64_to_expr(e.val)
            return expr
        
        elif eclass is IR.Add or eclass is IR.Mul:
            lhs  = self._compile_expr(e.lhs)
            rhs  = self._compile_expr(e.rhs)
            C_op = C.hwrap_add if eclass is IR.Add else C.hwrap_mul
            res  = C_op(lhs,rhs)
            return res
        
        elif eclass is IR.Sum:
            self._ctxt.push()
            # create rdom
            rb   = repr(e.idxname).encode('utf-8')
            lo   = C.hwrap_i32_to_expr(0)
            hi   = C.hwrap_i32_to_expr(e.range)
            rng  = (hw_expr_t * 2)(lo,hi)
            rdom = C.hwrap_new_rdom(rb,1,rng)
            # bind rdom in context as expr
            rexp = C.hwrap_rdom_to_expr(rdom)
            self._ctxt.set(e.idxname,rexp)
            # finally compile body...
            body = self._compile_expr(e.body)
            self._ctxt.pop()
            
            res  = C.hwrap_big_sum(rdom,body)
            return res
        
        elif eclass is IR.Access:
            return self._compile_access(e)
        
        elif eclass is IR.Indicate:
            li      = self._ctxt.get(e.arg.lhs)
            ri      = self._ctxt.get(e.arg.rhs)
            assert type(li) == hw_expr_t
            assert type(ri) == hw_expr_t
            
            eq      = C.hwrap_eq(li,ri)
            body    = self._compile_expr(e.body)
            zero    = C.hwrap_f64_to_expr(0.0)
            res     = C.hwrap_select(eq, body, zero)
            return res
            
        else: assert False, "unrecognized IR case"
    
    def _compile_access(self,e):
        idxs    = []
        while type(e) is IR.Access:
            # note that we pull off accesses right-to-left
            idxs.insert(0,e.idx)
            e = e.base
        assert type(e) is IR.Var
        F       = self._compile_expr(e)
        assert type(F) is hw_func_t
        
        # lookup index expressions, and create access
        n_idx   = len(idxs)
        exprs   = [ self._ctxt.get(i)
                    for i in idxs ]
        idx_arr = (hw_expr_t * n_idx)(*exprs)
        a   = C.hwrap_access_func(F,n_idx,idx_arr)
        return a

    def run(self,inputs,output=None):
        # check and bind input arguments
        hbufs = []
        for nm in self._arg_params: # that all inputs are defined
            if not nm in inputs:
                raise TypeError(f"expected input argument '{nm}'")
        for nm,val in inputs.items():
            if not nm in self._arg_typs:
                raise TypeError(f"unexpected input, named '{nm}'")
            T = self._arg_typs[nm]
            _type_value_match(T,val)
            P = self._arg_params[nm]
            if T is IR.tnum: # scalar
                assert type(P) is hw_param_t
                C.hwrap_set_param(P, ctypes.byref(ctypes.c_double(val)))
            else:
                assert type(P) is hw_img_t
                hbuf = _ndarray_to_halide_buf(val)
                hbufs.append(hbuf) # prevent early de-allocation
                C.hwrap_set_img(P, ctypes.byref(hbuf))
        
        # handle unsupplied output...
        if output is None:
            typ = self._ret_typ
            if typ is IR.tnum: # scalar case
                output = np.array([0.0])
            else: # tensor case
                shape = []
                while type(typ) is IR.TTensor:
                    shape.append(typ.range)
                    typ = typ.typ
                assert typ is IR.tnum
                
                output = np.ndarray(dtype='double', shape=shape, order='F')
        # check output
        if self._ret_typ is IR.tnum:
            if (type(output) != np.ndarray or
                len(output.shape) != 1 or
                output.shape[0] != 1):
                    raise TypeError("Expected numpy.ndarray of shape [1]")
        else:
            _type_value_match(self._ret_typ, output)
        # bind output
        outbuf  = _ndarray_to_halide_buf(output)
        hbufs.append(outbuf)
        
        # make sure the pipeline is auto-scheduled
        if self._first_run:
            self._first_run = False
            #C.hwrap_autoschedule_func( self._ret_Func )
            
        # run the pipeline
        C.hwrap_realize_func( self._ret_Func, outbuf )
        
        # potentially extract the output
        return_val = output
        if self._ret_typ is IR.tnum:
            return_val = output[0]
        return return_val

    


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #




# wrap type-checking in an object that we can use to keep track of
# the arguments etc. with
class TypedFunc:
    def __init__(self,argtyps,expr):
        self._orig_expr = expr
        
        # re-pack the args a few ways
        self._arglist = []
        self._argsyms = {}
        self._symtyps = {}
        assert type(argtyps) is list
        for nm,typ in argtyps:
            self._arglist.append( (nm,typ) )
            s = Sym(nm)
            self._argsyms[nm] = s
            self._symtyps[nm] = (s,typ)
        
        # run type-checking to get the modified IR
        self._expr    = TypeChecker(expr,self._symtyps).get_out_ir()

    def __str__(self):
        args = ", ".join([ f"{st[0]}:{st[1]}"
                           for n,st in self._symtyps.items() ])
        return (f"Function({args}) : {self._expr.typ}\n"
                f"    {self._expr.__str__(0,'    ')}")

def _TypedFunc_eliminate_pairs(self):
    e       = self._expr
    atyps   = self._symtyps
    self._expr = PairEliminator(e,atyps,e.typ).get_result()
TypedFunc.eliminate_pairs = _TypedFunc_eliminate_pairs

def _TypedFunc_lift_lets(self):
    e       = self._expr
    atyps   = self._symtyps
    self._expr = LetFlatten(e,atyps,e.typ).get_result()
TypedFunc.lift_lets = _TypedFunc_lift_lets

def _TypedFunc_gen_normalize(self):
    e       = self._expr
    atyps   = self._symtyps
    self._expr = GenNormalize(e,atyps,e.typ).get_result()
TypedFunc.gen_normalize = _TypedFunc_gen_normalize

def _TypedFunc_compile(self):
    # protect against re-compilation
    try:
        getattr(self,'_compiled_obj')
        return
    except: pass
    # run normalization
    self.eliminate_pairs()
    self.lift_lets()
    self.gen_normalize()
    # do the compilation via the CodeGen object
    e       = self._expr
    atyps   = self._symtyps
    CG      = Halide_CodeGen(e,atyps,e.typ)
    self._compiled_obj = CG

def _TypedFunc_run(self,inputs,output=None):
    self.compile()
    # actual execution
    return self._compiled_obj.run(inputs,output)

TypedFunc.compile   = _TypedFunc_compile
TypedFunc.run       = _TypedFunc_run



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #




def _Func_jit_compile(self):
    try: getattr(self,'_typed_func'); return
    except: pass
    
    self._typed_func = TypedFunc(self._arglist,self._expr)
    self._typed_func.compile()

def Func_jit_exec(self, *args, **kwargs):
    self._jit_compile()
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
            call_args[nm] = kwargs[nm]

    # then fill in the remainder with the unnamed arguments
    arg_i   = 0
    for nm,typ in self._arglist:
        if not nm in call_args:
            assert(arg_i < len(args))
            val     = args[arg_i]
            arg_i   = arg_i + 1
            call_args[nm] = val

    # finally, execute
    return self._typed_func.run(call_args)

Func._jit_compile = _Func_jit_compile
Func.jit_exec = Func_jit_exec

def _Func_str_maybe(self):
    self._jit_compile()
    return str(self._typed_func)
Func.__str__ = _Func_str_maybe




















n       = 3
R       = IR.tnum
Rn      = IR.TTensor(n,R)
Rnn     = IR.TTensor(n,Rn)
Rnnn    = IR.TTensor(n,Rnn)
x       = IRv0.Var('x')
A, D    = IRv0.Var('A'), IRv0.Var('D')
a, b, c = IRv0.Var('a'), IRv0.Var('b'), IRv0.Var('c')
i, j, k = 'i', 'j', 'k'

store_order = 'F'
xv = np.array([4.,7.,1.], order=store_order)
Av = np.array([[5.,2.,0.],[2.2,0.,4.5],[0.,6.1,3.3]], order=store_order)
av = np.array([9.2,5.4,7.1], order=store_order)
bv = np.array([0.3,2.1,1.6], order=store_order)
cv = np.array([0.,0.,1.], order=store_order)
Dv = np.array([[[ 1., 2., 3.],[ 4., 5., 6.],[ 7., 8., 9.]],
               [[10.,11.,12.],[13.,14.,15.],[16.,17.,18.]],
               [[19.,20.,21.],[22.,23.,24.],[25.,26.,27.]]], order=store_order)

pp2     = IRv0.Pair
p0      = lambda x: IRv0.Proj(0, x)
p1      = lambda x: IRv0.Proj(1, x)
let     = lambda x,r,b: IRv0.Let(x,r,b)
Gen     = lambda i,r,e: IRv0.Gen(i,r,e)
Sum     = lambda i,r,e: IRv0.Sum(i,r,e)


Axc     = Gen(i,n, c[i] + Sum(j,n, A[i,j] * x[j] ))
tr      = Sum(i,n, A[i,i])
diag    = Gen(i,n, Gen(j,n, IRv0.Eq(i,j) * x[i] ))
ab3     = let('x', Gen(j,n, pp2( pp2(a[j], b[j]), a[j]*b[j] )),
              Sum(j,n, p0(p0(x[j])) + p1(p0(x[j])) + p1(x[j]) ))
ctrct   = Gen(i,n, Sum(j,n, let('x', Sum(k,n, D[i,j,k] * a[k]),
                                 x * b[j] )))
prod    = Gen(i,n, Sum(j,n, A[i,j] * x[j] ))
qprod   = Sum(k,n, prod[k] * x[k])
dprod   = Sum(j,n, Sum(i,n, a[j] * diag[j,i] * a[i] ))

FAxc    = Func([('A',Rnn),('x',Rn),('c',Rn)], Axc)
Ftr     = Func([('A',Rnn)], tr)
Fdiag   = Func([('x',Rn)], diag)
Fab3    = Func([('a',Rn),('b',Rn)], ab3)
Fctrct  = Func([('D',Rnnn),('a',Rn),('b',Rn)], ctrct)
Fqprod  = Func([('A',Rnn),('x',Rn)], qprod)
Fdprod  = Func([('x',Rn),('a',Rn)] ,dprod)

print(FAxc)
print(Ftr)
print(Fdiag)
print(Fab3)
print(Fctrct)
print(Fqprod)
print(Fdprod)


def test(name,F,*args):
    print('Run Test: ',name)
    print(*args)
    l_args = [ a.tolist() for a in args ]
    print('compiled:    ', F.jit_exec(*args))
    print('interpreted: ', F.interpret(*l_args))

test("Axc",FAxc,Av,xv,cv)
test("tr",Ftr,Av)
test("diag",Fdiag,xv)
test("ab3",Fab3,av,bv)
test("ctrct",Fctrct,Dv,av,bv)
test("qprod",Fqprod,Av,xv)
test("dprod",Fdprod,xv,av)






