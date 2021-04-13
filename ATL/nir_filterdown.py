
from .prelude import *

from . import atl_types as T

from fractions import Fraction
from collections import namedtuple, defaultdict

import math

from .norm_ir import NIR, NIR_Stmts, nir_mul


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def ibrange(ranges,base=0):
  return [ NIR.idx_bind(NIR.ivar(i+base), r) for i,r in enumerate(ranges) ]
def eq_list(N, offset=0):
  return [ NIR.Alias(NIR.ivar(i), NIR.ivar(N+i+offset)) for i in range(N) ]
def typ_binds(typ):
  shape = typ.shape_or_scalar()
  return ( ibrange(shape), ibrange(shape,len(shape)), eq_list(len(shape)) )

def cache_dict():
  return defaultdict(cache_dict)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class NIR_FilterDown:
  """ Pass to take normalized-IR code and push sums and predicates towards
      the leaves as much as possible.

      Call ``result()`` to get the transformed NIR.

      Parameters
      ==============
      nir : NIR.function
        function to apply push-down to
  """
  def __init__(self, nir):
    self._nir   = nir

    self.cache  = cache_dict()

    def rec(out):
      if type(out) is NIR.OutLeaf:
        shape = out.expr.type.shape_or_scalar()
        expr  = self.pushdown( out.expr, [ False for _ in shape ], [] )
        expr  = expr.simplify()
        return NIR.OutLeaf( expr, out.type, out.srcinfo )
      elif type(out) is NIR.Tuple:
        return NIR.Tuple( [ rec(a) for a in out.args ],
                            out.type, out.srcinfo )
      else: assert False, "bad case"
    output = rec(nir.output)

    self.final = NIR.function( nir.name, nir.arg_order, nir.rettype,
                               nir.vars, nir.sizes, nir.relations,
                               output, nir.srcinfo )

  def result(self):
    return self.final

  def pushdown(self, e, sum_idx, preds):
    lookup = self.cache[e][tuple(sum_idx)][tuple(preds)]
    if type(lookup) is defaultdict:
      lookup = self.pushdown_helper(e, sum_idx, preds)
      self.cache[e][tuple(sum_idx)][tuple(preds)] = lookup
    return lookup

  def pushdown_helper(self, e, sum_idx, preds):
    """ sum_idx is a [bool] array that has one entry
        for every dimension of e.type.  If the array is True in an entry,
        then the pushdown should effect a big summation over that index

        preds is a list (implicit conjunction) of predicates over the
        indices [0,...,N-1] where N is the number of dimensions of e.type.
        preds captures filtering relationships between
        these indices/dimensions

        That is, we're attempting to rewrite/simplify a compound
        expression of the form:
            SUM[idx] [preds] * e
    """
    etyp  = type(e)
    shape = e.type.shape_or_scalar()
    assert len(sum_idx) == len(shape)
    do_wrap = any(sum_idx) or len(preds) > 0

    # push down through all parts of a simple summation...
    if etyp is NIR.Sum:
      terms     = []
      if e.offset != 0.0:
        terms.append( self.pushdown( NIR.Const(e.offset, e.type),
                                     sum_idx, preds ) )
      for t in e.terms:
        terms.append( self.pushdown( t, sum_idx, preds ) )
      return NIR.Sum(0.0, terms, terms[0].type).simplify()

    else:
      # a helper function that filters down the sum_idx
      # to a set of summations s.t. the joint summation is over
      # exactly 0 or 1 settings of the indices.
      #   e.g. SUM[i0,i1] [i0 == i1] * e
      #     can be filtered down to only i0, which can be pushed down
      #     through non-linear functions safely
      #        SUM[i1] (SUM[i0] [i0 == i1] * e)
      # lo is an offset to the indexing of sum_idx indices in preds
      def sums_at_most_once(sum_idx, preds, lo=0):
        sumlen = len(sum_idx)
        if sumlen == 0:
          return []
        # set this to True as we decide to use said summation
        sum_used = [ False for _ in sum_idx ]
        def chk(iv):
          i = iv.name - lo
          if 0 <= i < sumlen and sum_idx[i]:
            if not sum_used[i]:
              sum_used[i] = True
            return True
          return False
        for p in preds:
          ptyp    = type(p)
          if ptyp is NIR.Alias:
            if not chk(p.lvar):
              chk(p.rvar)
          elif ptyp is NIR.Cmp and p.op == '==':
            for a in p.eq.terms:
              if a.ivar is not None:
                if chk(a.ivar):
                  break
          else:
            pass

        return sum_used

      def any_summed(p, sum_idx, lo=0):
        def tst(i):
          ishift = i.name-lo
          return 0 <= ishift < len(sum_idx) and sum_idx[ishift]
        ptyp      = type(p)
        if ptyp is NIR.Alias:
          return tst(p.lvar) or tst(p.rvar)
        elif ptyp is NIR.Cmp:
          ivars = [ a.ivar for a in p.eq.terms if a.ivar is not None ]
          return any( tst(i) for i in ivars )
        elif ptyp is NIR.Relation:
          return any( tst(a) for a in p.args )
        elif ptyp is NIR.Conj or ptyp is NIR.Disj:
          return any( any_summed(pp,sum_idx,lo) for pp in p.preds )
        else:
          return False


      # if this is a point-wise function, then we can't:
      #   (a) push the sums through it without linearity
      #       (caveat: or the guarantee that the sum is over at most
      #                one non-zero value, which is a more important case)
      #   (b) push the predicates through it without
      #       a guarantee that zero-valued inputs will not cause crashes
      # We will try to propagate predicates, but wrap this in a summation
      # regardless.
      if etyp is NIR.BuiltIn or etyp is NIR.Pow:
        if etyp is NIR.Pow:
          assert e.power != 1 and e.power != 0
          crash_on_zero = e.power < 0
        else:
          crash_on_zero = e.f.crash_on_zero()

        if crash_on_zero:
          # don't push any predicates or sums down
          sum_pushed  = [ False for _ in sum_idx ]
          arg_preds   = []
        else:
          # otherwise, we can push-down all of the predicates
          # and can sum any index where we are guaranteed that the
          # summation will be over at most 1 non-zero value
          sum_pushed  = sums_at_most_once(sum_idx, preds)
          arg_preds   = preds

        if etyp is NIR.Pow:
          base  = self.pushdown( e.base, sum_pushed, arg_preds )
          e     = NIR.Pow(base, e.power, base.type)
          shape = base.type.shape_or_scalar()
        else:
          args  = [ self.pushdown( a, sum_pushed, arg_preds ) for a in e.args ]
          e     = NIR.BuiltIn(e.f, args, args[0].type)
          shape = args[0].type.shape_or_scalar()

        # account for any tensor indices that we've eliminated via summation
        sum_idx = [ si for si,sp in zip(sum_idx,sum_pushed) if not sp ]
        preds   = [ p for p in preds if not any_summed(p, sum_pushed) ]
        do_wrap = any(sum_idx) or len(preds) > 0


      # Construct the term SUM[idx] [preds] * e
      # explicitly, and apply simplification to it.
      # the result is wrap_e
      wrap_e  = e
      if do_wrap:
        # shift the indices so that the following mapping holds
        #               old_gens
        #            |------------|
        # |----------|------------|
        #   new_gens      idxs
        n_gen   = len(sum_idx) - sum(sum_idx)
        remap   = Context()
        for i,_ in enumerate(sum_idx):
          remap.set(NIR.ivar(i),NIR.ivar(i+n_gen))
        unsummed = [ i for i,dosum in enumerate(sum_idx) if not dosum ]
        preds   = ([ p.subst(remap) for p in preds ] +
                   [ NIR.Alias( NIR.ivar(i), NIR.ivar(n_gen+j) )
                     for i,j in enumerate(unsummed) ])

        # new bindings
        sum_rng = list(zip(sum_idx, e.type.shape_or_scalar()))
        gens    = ibrange([ rng for dosum,rng in sum_rng if not dosum ])
        idxs    = ibrange([ rng for _,rng in sum_rng ], base=n_gen)
        typ     = T.tensor_shape([ ib.range for ib in gens ])

        wrap_e  = NIR.Contract(gens, [], 1.0, preds,
                               [NIR.factor(e,idxs,Fraction(1))],
                               typ).simplify()

      # if we were at a leaf before wrapping, then this is it; we're done
      if etyp is NIR.Var or etyp is NIR.Const:
        return wrap_e

      # if we were at a function before wrapping then we already
      # processed everything downstream...
      elif etyp is NIR.BuiltIn:
        return wrap_e

      # Only remaining possibilities are that we started with a
      # contraction or Pow before wrapping.  However, we might have
      # simplified away all of our summation and predicates, in which
      # case we should re-start push-down anew on this sub-expression
      e = wrap_e
      if type(e) is NIR.Pow and etyp is NIR.Pow:
        # prevent endless recursion
        return e
      elif type(e) is not NIR.Contract:
        shape = e.type.shape_or_scalar() # not sure why, but this fixed bug
        return self.pushdown( e, [ False for _ in shape ], [] )
      else:
        # otherwise, we must unpack the contraction, and thereby
        # generate new pushdown contexts for its factors

        # our task is to determine which predicates are local to
        # a given factor, and which indices for a given factor can
        # have summation pushed down or not.
        # Helper functions...
        def idx_range(fac):
          return ( (0,0) if len(fac.idx) == 0 else
                   (fac.idx[0].ivar.name, fac.idx[-1].ivar.name + 1) )
        def all_in_range(p, lo,hi):
          def tst(i):
            return lo <= i.name < hi
          ptyp      = type(p)
          if ptyp is NIR.Alias:
            return tst(p.lvar) and tst(p.rvar)
          elif ptyp is NIR.Cmp:
            ivars = [ a.ivar for a in p.eq.terms if a.ivar is not None ]
            return all( tst(i) for i in ivars )
          elif ptyp is NIR.Relation:
            return all( tst(a) for a in p.args )
          elif ptyp is NIR.Conj or ptyp is NIR.Disj:
            return all( all_in_range(pp,lo,hi) for pp in p.preds )
          else:
            return True
        # find a list (with possible duplicates) of all indices in the
        # lo,hi range that occur within predicate `p`
        def range_filter(p, lo,hi):
          res = []
          def chk(i):
            if lo <= i.name < hi:
              res.append(i.name)
          def rec(p):
            ptyp    = type(p)
            if ptyp is NIR.Alias:
              chk(p.lvar); chk(p.rvar)
            elif ptyp is NIR.Cmp:
              for a in p.eq.terms:
                if a.ivar is not None:
                  chk(a.ivar)
            elif ptyp is NIR.Relation:
              for a in p.args:
                chk(a)
            elif ptyp is NIR.Conj or ptyp is NIR.Disj:
              for pp in p.preds:
                rec(pp)
            else:
              pass
          rec(p)
          return res

        # extract information on predicates in the whole contraction
        max_idx     = (len(e.gen_binds) + len(e.sum_binds) +
                       sum([ len(f.idx) for f in e.factors ]))
        e_p         = NIR.Conj(e.preds)
        eqv_map     = e_p.eqv_map(max_idx, use_greatest = True)
        e_p         = e_p.subst(eqv_map).simplify()
        assert e_p != NIR.FalseP()
        if e_p == NIR.TrueP():
          e_preds   = []
        elif type(e_p) is NIR.Conj:
          e_preds   = e_p.preds
        else:
          e_preds   = [e_p]
        # note that e_preds does not contain the aliasing predicates
        # implied by eqv_map

        # compute list of lists representation of the aliasing partition
        eqv_classes = [ [] for _ in range(0,max_idx) ]
        for i in range(0,max_idx):
          eqv_classes[ eqv_map.get(NIR.ivar(i)).name ].append(i)
        eqv_classes = [ x for x in eqv_classes if len(x) > 0 ]

        # return sum_idx, pred compatible arrays
        def factor_local(fac, lo,hi):
          sum_idx   = [ True for _ in fac.idx ]
          preds     = []

          # process e_preds
          for p in e_preds:
            if all_in_range(p, lo,hi):
              preds.append(p)
            else:
              # if the predicate spans this factor
              # and non-factor indices, then the summation over
              # any factor indices (i.e. in-range) occuring in the predicate
              # cannot be pushed down
              for i in range_filter(p, lo,hi):
                sum_idx[ i - lo ] = False

          # handle aliasing equivalence classes
          for ec in eqv_classes:
            fec = [ i for i in ec if lo <= i < hi ]
            if len(fec) > 0:
              for i,j in zip(fec[:-1],fec[1:]):
                preds.append(NIR.Alias(NIR.ivar(i), NIR.ivar(j)))
              if len(fec) != len(ec):
                sum_idx[ fec[-1] - lo ] = False

          return sum_idx, preds

        # Now, we can execute the pushdown and re-assemble the contraction
        sums        = e.sum_binds
        factors     = []
        for f in e.factors:
          lo,hi           = idx_range(f)
          sum_idx, preds  = factor_local(f, lo,hi)

          # we now need to cope with the exponent present.
          if f.power == 1:
            pass
          else:
            # if the power is negative, then nothing is safe to
            # push-down without possibly causing a divide-by-zero bug
            if f.power < 0:
              sum_idx = [ False for _ in sum_idx ]
              preds = []
            else:
              # otherwise, we can safely push all predicates through
              # but not sums... UNLESS we can certify that the
              # predicate/summation combo sums over exactly 0 or 1 elem...
              sum_idx = sums_at_most_once(sum_idx, preds, lo)

          # whether we have the original set of predicates
          # from factor_local(), or filtered by need to push through
          # an exponent, we can now do the factor pushdown

          # remove the predicates we're pushing down from the
          # predicate list at this contraction.  We make a special
          # exception when none of the indices referred to in the predicate
          # are being pushed down as summations.  In that case,
          # we may be able to save work by both pushing the predicate
          # down and retaining it at this contraction.
          to_remove = [ p for p in preds if any_summed(p, sum_idx, lo) ]
          e_preds = [ p for p in e_preds if p not in to_remove ]

          # remap the predicate list
          pred_remap = Context()
          for k,ib in enumerate(f.idx):
            pred_remap.set(ib.ivar, NIR.ivar(k))
          preds   = [ p.subst(pred_remap) for p in preds ]

          # split the factor indices between those which will
          # be eliminated via summation and those which will be retained
          keep_idx  = []
          for ib,dosum in zip(f.idx, sum_idx):
            if dosum:
              # Note that we are only allowed to sum indices which
              # are only predicate-connected to other indices in the
              # factor scope
              pass
              #sums.append(ib)
            else:
              keep_idx.append(ib)

          f_e       = self.pushdown(f.expr, sum_idx, preds)
          factors.append(NIR.factor(f_e, keep_idx, f.power))

        # we've filtered down e_preds during the factor pushdowns,
        # but we still need to reincorporate some of the aliasing
        # relationships
        for ec in eqv_classes:
          if len(ec) > 0:
            i0 = ec[0]
            for i in ec[1:]:
              e_preds.append(NIR.Alias(NIR.ivar(i0),NIR.ivar(i)))

        return NIR.Contract( e.gen_binds, sums, e.coeff,
                             e_preds, factors, e.type ).simplify()
















