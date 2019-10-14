
from .prelude import *

from . import atl_types as T

from fractions import Fraction
from collections import namedtuple

import math

from .norm_ir import NIR



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

def sort_groups(lst):
  xs    = sorted(lst)
  grp   = []
  for x in xs:
    # first iteration
    if len(grp) == 0:
      grp.append(x)
    # x is not equal to elements of grp; group switch
    elif grp[-1] < x:
      yield grp
      grp = [x]
    # otherwise x is a member of the current group
    else:
      grp.append(x)
  # final group
  if len(grp) > 0:
    yield grp

# --------------------------------------------------------------------------- #

# supports substitution of ivars and size variables with constants
# with other variables and with other affine expressions
@extclass(NIR.affine)
def subst(a,ctxt):
  offset  = a.offset
  terms   = []
  for t in a.terms:
    key = t.ivar or t.size
    val = ctxt.get(key)
    if val is None: val = key
    vclass = type(val)
    if vclass is NIR.ivar:
      terms.append( NIR.aterm(t.coeff, val, None) )
    elif vclass is Sym:
      terms.append( NIR.aterm(t.coeff, None, val) )
    elif vclass is Fraction or vclass is int:
      offset += t.coeff * val
    elif vclass is NIR.affine:
      offset += t.coeff * val.offset
      for vt in val.terms:
        terms.append(NIR.aterm( t.coeff*vt.coeff, vt.ivar, vt.size ))
    else: assert False, "unexpected substitution case"
  return NIR.affine(offset, terms)

# canonical ordering...
@extclass(NIR.ivar)
def __lt__(lhs,rhs):
  return lhs.name < rhs.name

@extclass(NIR.aterm)
def __lt__(lhs,rhs):
  if lhs.ivar:
    if rhs.ivar:  return lhs.ivar < rhs.ivar
    else:         return True
  else:
    if rhs.size:  return lhs.size < rhs.size
    else:         return False

@extclass(NIR.affine)
def simplify(a):
  if hasattr(a,'_simplified'): return a._simplified
  terms     = []
  for ts in sort_groups(a.terms):
    coeff   = sum([ t.coeff for t in ts ])
    if coeff != 0:
      terms.append(NIR.aterm( coeff, ts[0].ivar, ts[0].size ))
  a._simplified = NIR.affine(a.offset, terms)
  return a._simplified

# --------------------------------------------------------------------------- #

@extclass(NIR.pred)
def subst(p,ctxt):
  pclass = type(p)
  if   pclass is NIR.Alias:
    i, j  = ctxt.get(p.lvar), ctxt.get(p.rvar)
    i     = p.lvar if i is None else i
    j     = p.rvar if j is None else j
    return NIR.Alias(i,j)

  elif pclass is NIR.Cmp:
    return NIR.Cmp(p.op, p.eq.subst(ctxt))

  elif pclass is NIR.Relation:
    args      = [ ctxt.get(a) for a in p.args ]
    args      = [ (a if i is None else i) for i,a in zip(args,p.args) ]
    return NIR.Relation(p.name,args)

  elif pclass is NIR.Conj or pclass is NIR.Disj:
    return pclass( [ subp.subst(ctxt) for subp in p.preds ] )

  elif pclass is NIR.TrueP or pclass is NIR.FalseP:
    return p

  else: assert False, "unexpected case"

@extclass(NIR.pred)
def simplify(p):
  if hasattr(p, '_simplified'): return p._simplified
  pclass      = type(p)
  if   pclass is NIR.Alias:
    i,j       = p.lvar, p.rvar
    if j < i: i,j = j,i
    if i == j:  psimpl = NIR.TrueP()
    else:       psimpl = NIR.Alias(i,j)

  elif pclass is NIR.Cmp:
    eqsimpl   = p.eq.simplify()
    terms     = eqsimpl.terms
    # case: equation is a constant
    if len(terms) == 0:
      c       = eqsimpl.offset
      if   p.op == "==": psimpl = NIR.TrueP() if c == 0 else NIR.FalseP()
      elif p.op == ">":  psimpl = NIR.TrueP() if c > 0  else NIR.FalseP()
      elif p.op == ">=": psimpl = NIR.TrueP() if c >= 0 else NIR.FalseP()
      else: assert False, f"unrecognized op '{p.op}'"
    else:
      psimpl  = None
      if p.op == "==":
        # in this case, there is an extra ambiguity in the affine expression,
        # since  -eqsimpl == 0  iff.  eqsimpl == 0.
        # This ambiguity is normalized by forcing the first term of
        # eqsimpl to have a positive coefficient
        if terms[0].coeff < 0:
          terms   = [ NIR.aterm( -t.coeff, t.ivar, t.size ) for t in terms ]
          eqsimpl = NIR.affine( -eqsimpl.offset, terms )

        # Also, try converting this equality to a recognized Alias
        if ( eqsimpl.offset == 0 and len(terms) == 2 and
             terms[0].coeff == 1 and terms[0].ivar and
             terms[1].coeff == -1 and terms[1].ivar ):
          assert terms[0].ivar < terms[1].ivar
          psimpl = NIR.Alias( terms[0].ivar, terms[1].ivar )

      # generic fall-through
      if not psimpl:
        psimpl  = NIR.Cmp(p.op,eqsimpl)

  elif pclass is NIR.Relation or pclass is NIR.TrueP or pclass is NIR.FalseP:
    psimpl    = p

  elif pclass is NIR.Conj or pclass is NIR.Disj:
    preds     = []
    collapse  = None
    for sub in p.preds:
      simp    = sub.simplify()
      if simp == NIR.TrueP():
        if pclass is NIR.Disj:  collapse = True
      elif simp == NIR.FalseP():
        if pclass is NIR.Conj:  collapse = False
      elif type(simp) == pclass:
        preds += simp.preds
      else:
        preds.append(simp)
    if collapse is True or (len(preds) == 0 and pclass is NIR.Conj):
      psimpl  = NIR.TrueP()
    elif collapse is False or (len(preds) == 0 and pclass is NIR.Disj):
      psimpl  = NIR.FalseP()
    else:
      # remove duplicates and canonicalize the predicate order
      psimpl  = pclass([ ps[0] for ps in sort_groups(preds) ])

  else: assert False, "unexpected case"

  p._simplified = psimpl
  return psimpl

pclass_order = {
  NIR.TrueP     : 0,
  NIR.FalseP    : 1,
  NIR.Alias     : 2,
  NIR.Cmp       : 3,
  NIR.Relation  : 4,
  NIR.Conj      : 5,
  NIR.Disj      : 6,
}

@extclass(NIR.pred)
def __lt__(lhs,rhs):
  assert isinstance(rhs, NIR.pred)
  pclass      = type(lhs)
  ordL, ordR  = pclass_order[pclass], pclass_order[type(rhs)]
  if   ordL < ordR: return True
  elif ordL > ordR: return False
  # otherwise we have matching class...
  if   pclass is NIR.TrueP or pclass is NIR.FalseP:
    return False # they're equal
  elif pclass is NIR.Alias:
    return (lhs.lvar,lhs.rvar) < (rhs.lvar,rhs.rvar)
  elif pclass is NIR.Cmp:
    return (lhs.op,id(lhs.eq)) < (rhs.op,id(rhs.eq))
  elif pclass is NIR.Relation:
    return (lhs.name,lhs.args) < (rhs.name,rhs.args)
  elif pclass is NIR.Conj or pclass is NIR.Disj:
    return [ id(p) for p in lhs.preds ] < [ id(p) for p in rhs.preds ]
  else: assert False, "unexpected case"

# this mapping will map every ivar in range(0,counter)
# to the least equivalent ivar
@extclass(NIR.pred)
def eqv_map(p,counter):
  # initial, no-info equivalence mapping
  emap        = Context()
  for i in range(0,counter):
    emap.set(NIR.ivar(i),NIR.ivar(i))

  # see wikipedia, path-halving
  def find(i):
    nonlocal emap
    parent    = emap.get(i)
    while parent != i:
      parpar  = emap.get(parent)
      emap.set(i, parpar)
      i       = parpar
      parent  = emap.get(i)
    return i

  # extract set of predicates to consider
  if type(p) is NIR.Conj:
    preds     = p.preds
  elif type(p) is NIR.Alias:
    preds     = [p]
  else:
    # nothing we can do with this...
    return emap

  for p in preds:
    pclass    = type(p)
    if pclass is NIR.Alias:
      #print(counter, p.lvar, p.rvar)
      x, y    = find(p.lvar), find(p.rvar)
      if y < x: x,y = y,x
      emap.set(y,x)
    else: pass # nothing to do with non-alias predicates...

  # compress the emap for instant lookups
  for i in range(0,counter):
    emap.set( NIR.ivar(i), find(NIR.ivar(i)) )
  return emap

# --------------------------------------------------------------------------- #

SUM_TERM = namedtuple('SUM_TERM', ['coeff','gen','sum','preds','factors'])
def as_SUM_TERM(e):
  eclass = type(e)
  if eclass is NIR.Contract:
    return SUM_TERM(e.coeff, tuple(e.gen_binds), tuple(e.sum_binds),
                             tuple(e.preds), tuple(e.factors))
  else:
    assert eclass is not NIR.Sum and eclass is not NIR.Const
    gen, out, eqs   = typ_binds(e.type)
    fac             = NIR.factor( e, out, Fraction(1) )
    return SUM_TERM( 1.0, tuple(gen), tuple(), tuple(eqs), (fac,) )

@extclass(SUM_TERM)
def __lt__(lhs,rhs):
  def idmap(xs): return tuple( id(x) for x in xs )
  return ( tuple( idmap(a) for a in lhs[1:] ) <
           tuple( idmap(b) for b in rhs[1:] ) )
del __lt__

@extclass(SUM_TERM)
def to_expr(t):
  typ = T.tensor_shape([ ib.range for ib in t.gen ])
  if t.coeff == 0.0:
    return NIR.Const( 0.0, typ )
  else:
    assert len(t.factors) > 0
    passthrough = ( t.coeff == 1.0 and
                    len(t.sum) == 0 and
                    len(t.factors) == 1 and
                    len(t.gen) == len(t.preds) and
                    len(t.gen) == len(t.factors[0].idx) and
                    t.factors[0].power == 1 )

    # if that all checks out, then check for
    # numbering lining up or not
    if passthrough:
      N = len(t.gen)
      for i,(g,o,p) in enumerate(zip(t.gen, t.factors[0].idx, t.preds)):
        if (g.ivar.name != i or
            o.ivar.name != i+N or
            type(p) != NIR.Alias or
            p.lvar.name != i or
            p.rvar.name != i+N):
          passthrough = False
          break

    # now dispatch based on whether we actually have a passthrough
    if passthrough:
      return t.factors[0].expr
    else:
      return NIR.Contract( list(t.gen), list(t.sum), t.coeff,
                           list(t.preds), list(t.factors), typ )

            # -------------------------------------------------- #

# once simplified, an expression may be...
#     NIR.Var
#     NIR.Const
#     NIR.Pow(b,p)        where b is not Const, and b is not Pow,
#                               p is not 0, and p is not 1
#     NIR.BuiltIn(f,as)   where as is not all Const
#     NIR.Sum(c,es)       where #es > 0, es contains no Sums and no Consts
#     NIR.Contract(gs,ss,c,ps,fs)
#                         where #fs > 0, c != 0, fs[i].pow != 0 and
#                               #(gs + ss + fs.idx) > 0, and
#                               fs contains no Consts, Pow,
#                                          nor Contract with f.power == 1
@extclass(NIR.expr)
def simplify(e):
  if hasattr(e, '_simplified'): return e._simplified
  eclass        = type(e)

  if eclass is NIR.Var or eclass is NIR.Const:
    esimpl      = e

  elif eclass is NIR.Pow:
    if e.power == 0:
      esimpl    = NIR.Const(1.0, e.type)
    else:
      base      = e.base.simplify()
      if e.power == 1:
        esimpl  = base
      elif type(base) is NIR.Const:
        val     = math.pow( base.val, float(e.power) )
        esimpl  = NIR.Const(val, e.type)
      elif type(base) is NIR.Pow:
        esimpl  = NIR.Pow( base.base, e.power * base.power )
      else:
        esimpl  = NIR.Pow(base, e.power, e.type)

  elif eclass is NIR.BuiltIn:
    args        = [ a.simplify() for a in e.args ]
    if all([ type(a) is NIR.Const for a in args ]):
      raise NotImplementedError('Const Propagation through BuiltIns')
    esimpl      = NIR.BuiltIn( e.f, args, e.type )

  elif eclass is NIR.Sum:
    # simplify and accumulate terms
    offset      = 0.0
    terms       = []
    for t in e.terms:
      t         = t.simplify()
      tclass    = type(t)
      if tclass is NIR.Const:
        offset += t.val
      elif tclass is NIR.Sum:
        offset += t.offset
        terms  += [ as_SUM_TERM(subt) for subt in t.terms ]
      else:
        terms.append( as_SUM_TERM(t) )

    # then we can process these terms to deduplicate them
    final_terms = []
    for ts in sort_groups(terms):
      coeff     = sum([ t.coeff for t in ts ])
      if coeff != 0.0:
        t       = SUM_TERM(coeff,ts[0].gen,ts[0].sum,
                                 ts[0].preds,ts[0].factors)
        final_terms.append(t.to_expr())

    if len(final_terms) == 0:
      esimpl    = NIR.Const( offset, e.type )
    elif len(final_terms) == 1 and offset == 0.0:
      esimpl    = final_terms[0]
    else:
      esimpl    = NIR.Sum( offset, final_terms, e.type )

  elif eclass is NIR.Contract:
    assert False, "Contract should dispatch to its own function"

  else: assert False, "unexpected case"

  e._simplified = esimpl
  return esimpl

            # -------------------------------------------------- #

C_FAC = namedtuple('C_FAC', ['expr','idx','power'])

@extclass(C_FAC)
def __lt__(lhs,rhs):
  cmap = lambda x: ( id(x.expr), tuple([ib.ivar for ib in x.idx]) )
  return cmap(lhs) < cmap(rhs)
del __lt__

@extclass(C_FAC)
def to_factor(f):
  return NIR.factor( f.expr, f.idx, f.power )
del to_factor

            # -------------------------------------------------- #

@extclass(NIR.Contract)
def simplify(e):
  if hasattr(e, '_simplified'): return e._simplified
  eclass        = type(e)

  remap         = Context()
  counter       = 1 + max( [ ib.ivar.name for ib in e.gen_binds ] +
                           [ ib.ivar.name for ib in e.sum_binds ] +
                           [ ib.ivar.name for f in e.factors
                                          for ib in f.idx ],
                           default = 0 )
  #print('\n', '*** SIMPL CONTRACT ***')
  #print(counter, len(e.gen_binds), len(e.sum_binds), len(e.factors) )
  #print([ ib.ivar.name for f in e.factors for ib in f.idx ])
  #print(e.factors)
  def fresh_binds(bds):
    nonlocal counter
    c           = counter
    counter     = c + len(bds)
    #print('fresh at ', counter, bds)
    new_bds     = ibrange([ b.range for b in bds ], base=c)
    for old,new in zip(bds,new_bds):
      remap.set(old.ivar, new.ivar)
    return new_bds

  coeff         = e.coeff
  factors       = []
  sums          = e.sum_binds.copy()
  preds         = e.preds.copy()

  # collect all simplified and flattened factors
  for f in e.factors:
    fe          = f.expr.simplify()
    feclass     = type(fe)
    power       = f.power
    idx         = f.idx
    #print("HERE", feclass)
    # collapse any exponents that we encounter
    if feclass is NIR.Pow:
      power    *= fe.power
      fe        = fe.base
      feclass   = type(fe)

    # now consider different cases...
    if power == Fraction(0):
      pass # don't add this to factors...
    elif feclass is NIR.Const:
      coeff    *= math.pow( fe.val, float(power) )
      # must make sure not to lose track of indices
      # we may have to analytically collapse them
      sums     += idx
    elif ( feclass is NIR.Var or
           feclass is NIR.BuiltIn or
           feclass is NIR.Sum ):
      factors.append(C_FAC( fe, idx, power ))
    elif feclass is NIR.Contract:
      if power != 1:
        factors.append(C_FAC( fe, idx, power ))
      # case where we merge the contractions...
      else:
        #print(fe)
        remap.push()
        coeff  *= fe.coeff
        for g,i in zip(fe.gen_binds, idx):
          remap.set(g.ivar, i.ivar)
        sums   += idx
        sums   += fresh_binds(fe.sum_binds)
        factors+= [ C_FAC( f.expr, fresh_binds(f.idx), f.power )
                    for f in fe.factors ]
        # remap the predicates using substitution
        preds  += [ p.subst(remap) for p in fe.preds ]
        remap.pop()
    else: assert False, "unhandled case"
  #print("HERELHKNSDOIFNS")

  # in order to eliminate as many non-generation, non-factor
  # index variables as possible, shuffle all `sum` variables to the
  # end using a renaming pass
  counter       = len(e.gen_binds) # reset counter
  remap.push()
  factors       = [ C_FAC(f.expr, fresh_binds(f.idx), f.power)
                    for f in factors ]
  sum_idx_0     = counter # remember where non-factor sums start
  sums          = fresh_binds(sums)
  preds         = [ p.subst(remap) for p in preds ]
  remap.pop()

  # next, simplify the system of predicates with a possible early exit
  conj_pred     = NIR.Conj(preds).simplify()

  # extract index equivalence classes from the
  # predicate system
  eqv_map       = conj_pred.eqv_map(counter)

  # we can perform these substitutions now to enable further simplifications
  # and improve grouping of factors
  #     note: many unnecessary sums are removed here
  sums          = [ s for s in sums if eqv_map.get(s.ivar) == s.ivar ]
  mapbind       = lambda idx: [ NIR.idx_bind( eqv_map.get(i.ivar), i.range )
                                for i in idx ]
  factors       = [ C_FAC(f.expr, mapbind(f.idx), f.power) for f in factors ]
  conj_pred     = conj_pred.subst(eqv_map).simplify()

  # group factors
  final_factors = []
  for fs in sort_groups(factors):
    power       = sum([ f.power for f in fs ])
    if power != 0.0:
      final_factors.append(C_FAC( fs[0].expr, fs[0].idx, power ))

  # and now unique-ify the index names into a canonical order
  #   NOTE: there are technical gaps to 100% formal canonicity
  remap.push()
  counter       = 0
  id_preds      = [] # additional predicates to record aliases

  for g in e.gen_binds:
    # capture aliases between generators
    lookup      = eqv_map.get(g.ivar)
    if lookup != g.ivar:
      id_preds.append(NIR.Alias(lookup,g.ivar))

    # allocate generator index names
    remap.set(g.ivar, g.ivar)
    counter    += 1
    
  # factors
  factors       = []
  for f in final_factors:
    idx         = []
    for i in f.idx:
      ib        = NIR.idx_bind( NIR.ivar(counter), i.range )
      # only remap the first occurence
      if remap.get(i.ivar) == None:
        remap.set(i.ivar, ib.ivar)
      counter  += 1
      idx.append(ib)

      lookup    = remap.get( eqv_map.get(i.ivar) )
      #print('fidx ', lookup.name, i.ivar.name, ib.ivar.name)
      if lookup != ib.ivar:
        id_preds.append(NIR.Alias(lookup,ib.ivar))

    factors.append(C_FAC(f.expr, idx, f.power).to_factor())

  # sum indices
  old_sums,sums = sums, []
  for s in old_sums:
    sb          = NIR.idx_bind( NIR.ivar(counter), s.range )
    # only remap the first occurence
    if remap.get(s.ivar) == None:
      remap.set(s.ivar, sb.ivar)
    counter    += 1

    lookup    = remap.get( eqv_map.get(s.ivar) )
    if lookup != sb.ivar:
      id_preds.append(NIR.Alias(lookup,sb.ivar))

  #print("ID", id_preds)
  conj_pred     = NIR.Conj([ conj_pred.subst(remap) ] + id_preds).simplify()
  #print('FINAL PRED', conj_pred)
  remap.pop()

  # predicate unpacking
  if   conj_pred == NIR.FalseP():
    e._simplified = NIR.Const( 0.0, e.type )
    return e._simplified
  elif conj_pred == NIR.TrueP():
    preds       = []
  elif type(conj_pred) == NIR.Conj:
    preds       = conj_pred.preds
  else:
    preds       = [conj_pred]

  # constant detection
  if (len(factors) == 0 and len(sums) == 0) or coeff == 0.0:
    e._simplified = NIR.Const( coeff, e.type )
    return e._simplified

  # pass through detection
  if len(sums) == 0 and len(factors) == 1 and coeff == 1.0:
    f = factors[0]
    if ( len(e.gen_binds) == len(preds) and
         len(e.gen_binds) == len(f.idx) and
         f.power == 1 ):
      N           = len(e.gen_binds)
      if all([
        ( g.ivar.name == i and o.ivar.name == i+N and
          type(p) == NIR.Alias and
          p.lvar.name == i and p.rvar.name == i+N )
        for i,(g,o,p) in enumerate(zip(e.gen_binds,f.idx,preds))
      ]):
        e._simplified = f.expr
        return e._simplified

  #print('*** END SIMPL CONTRACT ***\n')
  e._simplified = NIR.Contract( e.gen_binds,
                                sums,
                                coeff,
                                preds,
                                factors,
                                e.type )
  return e._simplified

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


