
from .prelude import *

from . import atl_types as T
from . import builtins  as B

from fractions import Fraction

from .frontend import AST
from .norm_ir import NIR, NIR_Check, NIR_Stmts

from math import gcd
def lcm(x,y): return x*y // gcd(x,y)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def ibrange(ranges,base=0):
  return [ NIR.idx_bind(NIR.ivar(i+base), r) for i,r in enumerate(ranges) ]
def eq_list(N, offset=0):
  return [ NIR.Alias(NIR.ivar(i), NIR.ivar(N+i+offset)) for i in range(N) ]
def typ_binds(typ):
  shape = typ.shape() if typ != T.num else []
  return ( ibrange(shape), ibrange(shape,len(shape)), eq_list(len(shape)) )

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class AST_to_NIR:
  def __init__(self, f, use_simplify=True):
    self._ctxt        = Context()
    self._idx_vars    = Context()
    self._relshapes   = {}
    self._ranges      = []
    self._ctxt.push()

    # handle arguments
    vs                = [ NIR.var_decl(v.name, v.type, v.srcinfo)
                          for v in f.vars ]
    szs               = [ NIR.size_decl(s.name, s.srcinfo)
                          for s in f.sizes ]
    rels              = [ NIR.rel_decl(r.name, r.sizes, r.srcinfo)
                          for r in f.relations ]
    for v in f.vars:
      self._ctxt.set( v.name, NIR.Var(v.name, [], v.type) )
    for r in f.relations:
      self._relshapes[r.name] = r.sizes

    # the body should be in a let-expression chain format.
    body              = f.body
    if type(body) is AST.Let:
      for s in body.stmts:
        rhs           = self.normalize(s.rhs).simplify()
        self._ctxt.set( s.name, rhs )
      body            = body.ret
    # now we have an expression which may be
    # a tuple in need of decomposition
    def decompose(e):
      if type(e) is AST.Tuple:
        args          = [ decompose(a) for a in e.args ]
        return NIR.Tuple( args, e.type, e.srcinfo )
      else:
        if e.type == T.num:
          enorm       = self.normalize(e)
        else:
          for r in e.type.shape():
            self._ranges.append(r)
          enorm       = self.normalize(e)
          self._ranges = []
        if use_simplify:
          enorm     = enorm.simplify()
        return NIR.OutLeaf( enorm, e.type, e.srcinfo )
    out = decompose(body)

    # finally, construct the normalized function object
    self._nir = NIR.function( name        = f.name,
                              arg_order   = f.arg_order,
                              rettype     = f.rettype,
                              vars        = vs,
                              sizes       = szs,
                              relations   = rels,
                              output      = out,
                              srcinfo     = f.srcinfo )
    NIR_Check(self._nir)

  def result(self):
    return self._nir

  def _ranges_type(self):
    typ = T.num
    for r in reversed(self._ranges):
      typ = T.Tensor(r,typ)
    return typ

  def get_pass_binds(self, typ):
    tranges     = typ.shape() if typ != T.num else []
    N_ctxt      = len(self._ranges)
    N           = N_ctxt + len(tranges)

    gen_binds   = ibrange(self._ranges)   + ibrange(tranges, base=N_ctxt)
    out_binds   = ibrange(self._ranges,N) + ibrange(tranges, base=N+N_ctxt)
    return gen_binds, out_binds, eq_list(N)

  def normalize(self, e):
    eclass  = type(e)
    if eclass is AST.Gen or eclass is AST.Sum:
      # these forms help scope the shape of their operand
      ibind           = NIR.idx_bind( NIR.ivar( len(self._ranges) ),
                                      e.range )

      self._ranges.append(e.range)
      self._idx_vars.push()
      self._idx_vars.set(e.name, ibind)
      result        = self.normalize(e.body)
      self._idx_vars.pop()
      self._ranges.pop()

      if eclass is AST.Gen:
        return result
      else:
        gen_binds, out_binds, preds = self.get_pass_binds(e.type)
        # need to sum using ibind, but numbered to come after 2*len(ranges)
        out_binds.append(NIR.idx_bind( NIR.ivar( 2*len(self._ranges) + 1 ),
                                       e.range ))

        return NIR.Contract( gen_binds, [], 1.0, preds,
                             [NIR.factor( result, out_binds, Fraction(1) )],
                             self._ranges_type() )

    elif (eclass is AST.Access or
          eclass is AST.Proj or
          eclass is AST.Var):
      return self.get_leaf_var(e)

    elif eclass is AST.Const:
      return NIR.Const(e.val, self._ranges_type())

    elif eclass is AST.BinOp:
      lhs         = self.normalize(e.lhs)
      rhs         = self.normalize(e.rhs)
      if   e.op == '+':
        return NIR.Sum(0.0, [lhs, rhs], lhs.type)
      elif e.op == '-':
        neg_one = NIR.Const( -1.0, self._ranges_type() )
        return NIR.Sum(0.0, [lhs, self.do_mul(neg_one,rhs)], lhs.type)
      elif e.op == '*':
        return self.do_mul(lhs, rhs)
      elif e.op == '/':
        return self.do_mul(lhs, NIR.Pow(rhs, Fraction(1), rhs.type))
      else: assert False, f"unrecognized op: {e.op}"

    elif eclass is AST.TensorLit:
      assert False, "TODO: handle tensor literals in normalized IR"
      # expand the tensor literal and continue...

    elif eclass is AST.BuiltIn:
      args = [ self.normalize(a) for a in e.args ]
      if e.f == B.sqrt:
        return NIR.Pow( args[0], Fraction(1,2), self._ranges_type() )
      else:
        return NIR.BuiltIn( e.f, args, self._ranges_type() )

    elif eclass is AST.Indicate:
      gen_binds, out_binds, preds = self.get_pass_binds(e.type)
      p, sum_binds                = self.norm_pred(e.pred, len(gen_binds)+
                                                           len(out_binds))
      body        = self.normalize(e.body)
      return NIR.Contract( gen_binds, sum_binds, 1.0, preds,
                           [NIR.factor(body, out_binds, Fraction(1))],
                           self._ranges_type() )

    else:
      assert eclass is not AST.Tuple
      assert eclass is not AST.Let
      assert False, "unrecognized AST case"

  # process
  #   AST.Access
  #   AST.Proj, and
  #   AST.Var nodes
  def get_leaf_var(self, e):
    eclass      = type(e)

    # collect together any accesses
    access      = []
    while eclass is AST.Access:
      access    = e.idx + access
      e, eclass = e.base, type(e.base)

    # collect together any projections
    projs       = []
    while eclass is AST.Proj:
      projs     = [e.idx] + projs
      e, eclass = e.base, type(e.base)

    assert eclass is AST.Var, "should be left with a variable"

    # get the variable
    x           = self._ctxt.get(e.name)

    # determine whether this variable is an input, and project if needed
    if type(x) is NIR.Var:
      # if it's a variable, then we can project
      assert len(x.proj) == 0
      typ       = x.type
      for i in projs:
        assert type(typ) is T.Tuple
        typ     = typ.types[i]
      x = NIR.Var( x.name, projs, typ )
    else:
      assert len(projs) == 0, "should not be projecting intermediates"

    # regardless, apply any necessary accesses.
    # NOTE: our context defines the expected shape to be a tensor of
    #       dimension [self._ranges]
    #     Therefore, if there are no accesses we must still broadcast
    #       the scalar value out to this dimensional tensor.  If there
    #       are accesses, then those accesses need to be converted
    #       into a general Contraction node.  Either way, we need to
    #       construct a Contraction node.
    gen_binds   = ibrange(self._ranges)
    out_binds   = ( ibrange(x.type.shape(), base=len(gen_binds))
                    if x.type != T.num else [] )
    preds       = [ self.name_aexpr( gi.ivar, self.norm_aexpr(a) )
                    for a,gi in zip(access,out_binds) ]

    # determine whether to return a contraction or not
    if len(gen_binds) + len(out_binds) == 0:
      return x
    else:
      return NIR.Contract( gen_binds, [], 1.0, preds,
                           [ NIR.factor(x,out_binds,Fraction(1)) ],
                           self._ranges_type() )

  def do_mul(self, lhs, rhs):
    rngs        = self._ranges
    N           = len(rngs)
    gen         = ibrange(rngs)
    l_out       = ibrange(rngs,base=N)
    r_out       = ibrange(rngs,base=2*N)
    preds       = eq_list(N) + eq_list(N,offset=N)
    assert lhs.type == rhs.type
    assert tuple(lhs.type.shape() if lhs.type != T.num else []) == tuple(rngs)

    return NIR.Contract( gen, [], 1.0, preds,
                         [ NIR.factor(lhs,l_out,Fraction(1)),
                           NIR.factor(rhs,r_out,Fraction(1)) ],
                         lhs.type )

  def norm_aexpr(self, a):
    aclass      = type(a)
    if aclass is AST.IdxConst:
      return NIR.affine( Fraction(a.val), [] )
    elif aclass is AST.IdxVar:
      name      = self._idx_vars.get(a.name).ivar
      return NIR.affine(Fraction(0),[ NIR.aterm(Fraction(1), name, None) ])
    elif aclass is AST.IdxSize:
      return NIR.affine(Fraction(0),[ NIR.aterm(Fraction(1), None, a.name) ])
    elif aclass is AST.IdxAdd:
      lhs       = self.norm_aexpr(a.lhs)
      rhs       = self.norm_aexpr(a.rhs)
      return NIR.affine(lhs.offset + rhs.offset, lhs.terms + rhs.terms)
    elif aclass is AST.IdxScale:
      idx       = self.norm_aexpr(a.idx)
      terms     = [ NIR.aterm(a.coeff * t.coeff, t.ivar, t.size)
                    for t in idx.terms ]
      return NIR.affine(a.coeff*idx.offset, terms)
    else: assert False, "unexpected case"

  # returns a pair of (pred, [idxbind])
  def norm_pred(self, p, count):
    pclass      = type(p)
    if pclass is AST.Cmp:
      lhs       = self.norm_aexpr(p.lhs)
      rhs       = self.norm_aexpr(p.rhs)
      op        = p.op
      if op == "<" or op == "<=":
        lhs, rhs  = rhs, lhs
        op = ">" if op == "<" else ">="
      # eq = lhs - rhs
      offset    = lhs.offset - rhs.offset
      terms     = lhs.terms.copy()
      for t in rhs.terms:
        terms.append(NIR.aterm( -1 * t.coeff, t.ivar, t.size ))
      return NIR.Cmp(op, NIR.affine(offset,terms)), []

    elif pclass is AST.Relation:
      R_shape   = self._relshapes[p.name]
      binds     = ibrange(R_shape, base=count)
      preds     = [ self.name_aexpr( i.ivar, self.norm_aexpr(a) )
                    for a,i in zip(p.args,binds) ]
      preds.append(NIR.Relation( p.name, [ i.ivar for i in binds ] ))
      return NIR.Conj(preds), binds

    elif pclass is AST.Conj or pclass is AST.Disj:
      lhs,lbind = self.norm_pred(p.lhs,count)
      count    += len(lbind)
      rhs,rbind = self.norm_pred(p.rhs,count)
      if pclass is AST.Conj:
        return NIR.Conj([lhs,rhs]), lbind+rbind
      else:
        return NIR.Disj([lhs,rhs]), lbind+rbind

    else: assert False, "unexpected case"

  def name_aexpr(self, ivar, rhs):
    terms       = rhs.terms + [ NIR.aterm(Fraction(-1), ivar, None) ]
    return NIR.Cmp("==", NIR.affine(rhs.offset, terms))

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class NIR_to_AST:
  def __init__(self, f):
    self._indices         = []
    self._idx_stack       = []

    used_varnames         = {}
    for sz in f.sizes:      used_varnames[str(sz.name)] = True
    for vd in f.vars:       used_varnames[str(vd.name)] = True
    for rd in f.relations:  used_varnames[str(rd.name)] = True
    self._used_varnames   = used_varnames
    self._count           = 0

    self._STMT            = NIR_Stmts(f.output)
    self._expr_cache      = {}
    self._stmts           = []

    self._vars            = {}
    for vd in f.vars:       self._vars[vd.name] = vd

    # process the body
    def recompose(out):
      if type(out) is NIR.Tuple:
        args              = [ recompose(a) for a in out.args ]
        return AST.Tuple( args, out.type, out.srcinfo )
      elif type(out) is NIR.OutLeaf:
        typ               = out.expr.type
        shape             = [] if typ == T.num else typ.shape()
        self.init_idx_stack(shape)
        return self.to_ast(out.expr)
      else: assert False, "unexpected case"
    body                  = recompose(f.output)

    if len(self._stmts) > 0:
      body                = AST.Let( self._stmts, body,
                                     body.type, body.srcinfo )

    self._result_ast = AST.function(
      name        = f.name,
      arg_order   = f.arg_order,
      rettype     = f.rettype,
      vars        = [ AST.var_decl(v.name, v.type, v.srcinfo)
                      for v in f.vars ],
      sizes       = [ AST.size_decl(s.name, s.srcinfo) for s in f.sizes ],
      relations   = [ AST.rel_decl(r.name, r.sizes, r.srcinfo)
                      for r in f.relations ],
      body        = body,
      srcinfo     = f.srcinfo,
    )

  def result(self):
    return self._result_ast

  def get_new_var(self):
    self._count      += 1
    name              = f"v_{self._count}"
    if name in self._used_varnames:
      return self.get_new_var()
    return Sym(name)

  def push_idx_binds(self):
    oldidx            = self._indices
    self._idx_stack.append(oldidx)
    self._indices     = [ ( AST.IdxVar( Sym(f"i{i}"), null_srcinfo() ), rng )
                          for i,(idx,rng) in enumerate(oldidx) ]

  def push_idx_copy(self):
    self._idx_stack.append([ (idx,rng) for idx,rng in self._indices ])

  def push_idx_frame(self, frame):
    self._idx_stack.append(self._indices)
    self._indices     = frame

  def init_idx_stack(self, ranges):
    self._idx_stack   = []
    self._indices     = [ ( AST.IdxVar( Sym(f"i{i}"), null_srcinfo() ), r )
                          for i,r in enumerate(ranges) ]

  def pop_idx_stack(self):
    self._indices     = self._idx_stack.pop()

  def wrap_gens(self, ast):
    assert ast.type is T.num
    for idx,rng in reversed(self._indices):
      assert type(idx) is AST.IdxVar
      ast             = AST.Gen( idx.name, rng, ast,
                                 T.Tensor(rng, ast.type), ast.srcinfo )
    return ast

  def wrap_access(self, ast):
    if ast.type == T.num:
      assert len(self._indices) == 0
      return ast
    else:
      shape           = ast.type.shape()
      assert tuple(shape) == tuple( rng for idx,rng in self._indices )
      return AST.Access( ast, [idx for idx,rng in self._indices],
                         T.num, null_srcinfo() )

  def to_ast(self, e, do_bind=False):
    if e in self._expr_cache:
      ast             = self._expr_cache[e]
    else:
      do_bind         = ( do_bind or
                          self._STMT.is_repeated(e) or
                          type(e) is NIR.Contract )
      if do_bind:
        self.push_idx_binds()
      ast             = self._to_ast_helper(e)
      if do_bind:
        ast           = self.wrap_gens(ast)
        self.pop_idx_stack()
        # variable
        varname       = self.get_new_var()
        # stmt
        stmt          = AST.assign( varname, ast.type, ast, null_srcinfo() )
        self._stmts.append(stmt)
        # var
        ast           = AST.Var( varname, ast.type, null_srcinfo() )
        self._expr_cache[e] = ast

    # regardless of whether this was memoized or computed fresh,
    # check if this is a leaf node that needs to be indexed.
    if type(ast) is AST.Var or type(ast) is AST.Proj:
      ast             = self.wrap_access(ast)

    return ast

  def _to_ast_helper(self, e):
    eclass = type(e)
    if eclass is NIR.Var:
      typ             = self._vars[e.name].type
      ast             = AST.Var(e.name, typ, null_srcinfo())

      for i in e.proj:
        typ           = typ.types[i]
        ast           = AST.Proj(i, ast, typ, null_srcinfo())

      return ast

    elif eclass is NIR.Const:
      return AST.Const( e.val, T.num, null_srcinfo() )

    elif eclass is NIR.Sum:
      off_ast         = AST.Const(e.offset, T.num, null_srcinfo())
      terms           = [] if e.offset == 0.0 else [ off_ast ]
      terms          += [ self.to_ast(t) for t in e.terms ]
      assert all([ t.type == T.num for t in terms ])

      ast             = terms[0]
      for t in terms[1:]:
        # get rid of the pattern ... + -1.0*t
        if ( type(t) is AST.BinOp and t.op == "*" and
             type(t.lhs) is AST.Const and t.lhs.val == -1.0 ):
          ast         = AST.BinOp("-", ast, t.rhs, T.num, null_srcinfo())
        else:
          ast         = AST.BinOp("+", ast, t, T.num, null_srcinfo())

      return ast

    elif eclass is NIR.Contract:
      return self._contract_to_ast(e)

    elif eclass is NIR.Pow:
      def is_small_int(p):
        p = abs(p)
        return p == 2 or p == 3 or p == 4
      if is_small_int(e.power):
        base        = self.to_ast(e.base, do_bind=True)
      else:
        base        = self.to_ast(e.base)

      assert base.type == T.num
      if e.power == 0:
        return AST.Const(1.0,T.num,null_srcinfo())

      elif e.power == 1:
        return base
      elif e.power == Fraction(1,2):
        return AST.BuiltIn( B.sqrt, [base], T.num, null_srcinfo() )
      elif e.power == -1:
        return AST.BinOp("/", AST.Const(1.0,T.num,null_srcinfo()),
                              base, T.num, null_srcinfo())
      elif e.power == Fraction(-1,2):
        return AST.BinOp("/", AST.Const(1.0,T.num,null_srcinfo()),
                              AST.BuiltIn( B.sqrt, [base] ),
                              T.num, null_srcinfo() )

      elif is_small_int(e.power):
        ast         = AST.BinOp("*", base, base, T.num, null_srcinfo())
        if abs(e.power) == -2: pass
        elif abs(e.power) == -3:
          ast = AST.BinOp("*", base, ast, T.num, null_srcinfo())
        elif abs(e.power) == -4:
          ast = AST.BinOp("*", ast, ast, T.num, null_srcinfo())
        if e.power > 0:
          return ast
        else:
          return AST.BinOp("/", AST.Const(1.0,T.num,null_srcinfo()),
                                ast, T.num, null_srcinfo())

      else:
        power       = AST.Const( float(e.power), T.num, null_srcinfo() )
        return AST.BuiltIn( B.pow, [base, power], T.num, null_srcinfo() )

    elif eclass is NIR.BuiltIn:
      args          = [ self.to_ast(a) for a in e.args ]
      return AST.BuiltIn( e.f, args, T.num, null_srcinfo() )

  def _contract_to_ast(self, e):
    # map generator variables to indexing expressions
    idx_map         = Context()
    for (idx,rng), g in zip(self._indices, e.gen_binds):
      assert rng == g.range
      idx_map.set(g.ivar, idx)

    # collect binding ranges for reference
    franges         = { i.ivar : i.range for f in e.factors
                                         for i in f.idx }
    for i in e.sum_binds:
      franges[i.ivar] = i.range

    # helper functions
    def mul_aff(s, eq):
      if s == 1:  return eq
      terms         = [ NIR.aterm(s*t.coeff, t.ivar, t.size)
                        for t in eq.terms ]
      return NIR.affine( s*eq.offset, terms )
    def rescale_aff(eq):
      scale         = eq.offset.denominator
      for t in eq.terms:
        scale       = lcm(scale, t.coeff.denominator)
      return mul_aff(scale, eq)

    # separate the predicates into parts:
    #   1. for filtering on
    #   2. for defining indices via substitutions
    conjp           = NIR.Conj(e.preds).simplify()
    preds           = conjp.preds if type(conjp) == NIR.Conj else [conjp]
    filters         = []
    substitutions   = []
    for p in preds:
      if type(p) is NIR.Alias:
        substitutions.append( NIR.affine(Fraction(0),[
            NIR.aterm( Fraction(1), p.lvar, None),
            NIR.aterm(Fraction(-1), p.rvar, None)
          ]))
      elif type(p) is NIR.Cmp:
        if p.op == ">" or p.op == ">=":
          filters.append(p)
        elif p.op == "==":
          substitutions.append( rescale_aff(p.eq) )
        else: assert False, "bad op"
      elif type(p) is NIR.TrueP: pass
      else:
        assert type(p) is NIR.Relation or type(p) is NIR.Disj
        filters.append(p)

    # now process the substitutions
    def term_is_free(t):
      return t.ivar != None and idx_map.get(t.ivar) == None
    subs            = [ (a, { t.ivar : True for t in a.terms
                              if term_is_free(t) })
                        for a in substitutions ]
    def split_eq(eq,iv):
      """ Split out the specified term from the equation """
      terms         = []
      ivterm        = None
      for t in eq.terms:
        if t.ivar == iv:  ivterm = t
        else:             terms.append(t)
      assert ivterm and len(terms) == len(eq.terms)-1
      return ivterm, NIR.affine(eq.offset, terms)
    def do_sub(iv,eq):
      ast_idx       = self.aff_to_idx(eq,idx_map)
      idx_map.set( iv, ast_idx )
      # clear out the free variable
      for a,FV in subs:
        if iv in FV: del FV[iv]
    sum_binds       = []
    def bind_idx(iv):
      sym           = Sym(f"i{iv.name}")
      sum_binds.append( (sym, franges[iv]) )
      idx_map.set( iv, AST.IdxVar(sym, null_srcinfo()) )
      # clear out the bound variable from free variables
      for a,FV in subs:
        if iv in FV: del FV[iv]

    # iterate until all substitutions have been processed
    while len(subs) > 0:
      keepsubs      = []
      did_sub       = False

      # try to find substitutions
      for eq,FV in subs:
        n_free      = len(FV)
        if n_free == 0:
          filters.append(NIR.Cmp("==",eq))
        elif n_free == 1: # try substitution
          freevar   = list(FV.keys())[0]
          freeterm, rest_eq = split_eq(eq, freevar)
          # do the substitution
          did_sub   = True
          if abs(freeterm.coeff) != 1:
            # in this case, substitute AND filter
            filters.append(NIR.Cmp("==",eq))
          # regardless substitute
          rest_eq   = mul_aff(freeterm.coeff, rest_eq)
          do_sub(freevar,rest_eq)
        else: # over 1 free variable...
          keepsubs.append( (eq,FV) )

      subs          = keepsubs

      # if we didn't do any substitutions, but
      # still have equations with free-variables, then we
      # need to pick some free variable to bind
      if not did_sub and len(subs) > 0:
        ivar        = None
        for eq,FV in subs:
          if len(FV) > 0:
            ivar    = list(FV.keys())[0]
            bind_idx(ivar)
            break

    # after processing all the substitutions,
    # bind any remaining sum variables or factor variables
    for i in e.sum_binds:
      if idx_map.get(i.ivar) == None:
        bind_idx(i.ivar)
    for f in e.factors:
      for i in f.idx:
        if idx_map.get(i.ivar) == None:
          bind_idx(i.ivar)

    # build factor product
    product         = None
    if e.coeff != 1.0:
      product       = AST.Const( e.coeff, T.num, null_srcinfo() )
    for f in e.factors:
      # re-bind the indices...
      self.push_idx_frame([ ( idx_map.get(i.ivar), i.range )
                            for i in f.idx ])
      # convert the factor to AST form
      fe            = f.expr
      if f.power != 1:
        fe          =  NIR.Pow(fe, f.power, f.expr.type)
      f_ast         = self.to_ast(fe)
      self.pop_idx_stack()
      # accumulate into the product
      if not product:
        product     = f_ast
      else:
        product     = AST.BinOp( "*", product, f_ast,
                                 T.num, null_srcinfo() )
    # edge case fix-up
    if not product:
      product       = AST.Const( 1.0, T.num, null_srcinfo() )

    # coming back up, we can now convert the filters if present
    # and wrap the product with an indicator
    # process the filters into a single predicate
    if len(filters) > 0:
      if len(filters) == 1:
        conjp       = filters[0]
      else:
        conjp       = NIR.Conj(filters)
      ast_p         = self.pred_to_ast(conjp, idx_map)
      product       = AST.Indicate( ast_p, product, T.num, null_srcinfo() )

    # finally, wrap with any summation bindings
    for sym,rng in reversed(sum_binds):
      product       = AST.Sum( sym, rng, product, T.num, null_srcinfo() )

    return product

  def pred_to_ast(self, p, idx_map):
    pclass          = type(p)
    assert pclass is not NIR.TrueP and pclass is not NIR.FalseP
    if pclass is NIR.Alias:
      return AST.Cmp( "==", idx_map.get(p.lvar), idx_map.get(p.rvar),
                      null_srcinfo() )
    elif pclass is NIR.Cmp:
      return AST.Cmp( p.op, self.aff_to_idx(p.eq, idx_map),
                      AST.IdxConst(0,null_srcinfo()), null_srcinfo() )
      pass
    elif pclass is NIR.Relation:
      args          = [ self.pred_to_ast(subp, idx_map) for subp in p.preds ]
      return AST.Relation(p.name, args, null_srcinfo())
    elif pclass is NIR.Conj or pclass is NIR.Disj:
      astOp         = AST.Conj if pclass is NIR.Conj else AST.Disj
      out           = self.pred_to_ast(p.preds[0], idx_map)
      for subp in p.preds[1:]:
        out         = astOp(out, self.pred_to_ast(subp, idx_map),
                            null_srcinfo())
      return out
    else: assert False, "unexpected case"

  def aff_to_idx(self, a, idx_map):
    idx             = None
    if a.offset != 0:
      if a.offset.denominator == 1:
        idx         = AST.IdxConst(a.offset.numerator, null_srcinfo())
      else:
        idx         = AST.IdxScale(a.offset, AST.Const(1,null_srcinfo()),
                                             null_srcinfo())
    for t in a.terms:
      if t.ivar:
        ast         = idx_map.get(t.ivar)
        assert ast, "expected all index variables to resolve"
      else:
        ast         = AST.IdxSize(t.size, null_srcinfo())

      if t.coeff != 1:
        ast         = AST.IdxScale(t.coeff, ast, null_srcinfo())
      idx           = ast if not idx else AST.IdxAdd(idx, ast,
                                                     null_srcinfo())
    if not idx: idx = AST.IdxConst(0, null_srcinfo())
    return idx














