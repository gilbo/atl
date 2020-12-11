
from .prelude import *

from . import atl_types as T

from fractions import Fraction
from collections import namedtuple, defaultdict
import itertools

import math

from .norm_ir import NIR, NIR_Stmts, nir_mul


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def cache_dict():
  return defaultdict(cache_dict)

PlanNode = namedtuple('PlanNode',['left','right'])
PlanLeaf = namedtuple('PlanLeaf',['obj','idx','summed','cost'])

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class NIR_Factorize:
  """ Pass to take normalized-IR code and factor apart products.
      The resulting NIR expression will not be simplified, so
      this should be called right before converting back to the AST

      Call ``result()`` to get the transformed NIR.

      Parameters
      ==============
      nir : NIR.function
        function to apply factorize to
  """
  def __init__(self, nir):
    self._nir   = nir

    self.cache  = cache_dict()

    def rec(out):
      if type(out) is NIR.OutLeaf:
        shape = out.expr.type.shape_or_scalar()
        expr  = self.factorize( out.expr )
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

  # how to estimate sizes for the purposes of costing factorization
  # plans
  def estimate(rng):
    if is_pos_int(rng):
      return rng
    else:
      return 100 # dummy value; can patch with other estimates if needed

  def factorize(self, e):
    lookup = self.cache[e][tuple(sum_idx)][tuple(preds)]
    if type(lookup) is defaultdict:
      lookup = self.factorize_helper(e, sum_idx, preds)
      self.cache[e][tuple(sum_idx)][tuple(preds)] = lookup
    return lookup

  def factorize_helper(self, e):
    """ ...
    """

    etyp  = type(e)

    # do nothing at leaf nodes
    if etyp is NIR.Var or etyp is NIR.Const:
      return e

    # in these cases, we want to factorize inside the
    # term, but can basically ignore this term otherwise...
    # (CAVEAT: some of the built-ins may support some kind of
    #          factorization...)
    elif etyp is NIR.Pow:
      base = self.factorize(e.base)
      return NIR.Pow(base, e.power, e.type)

    elif etyp is NIR.BuiltIn:
      args = [ self.factorize(a) for a in e.args ]
      return NIR.BuiltIn(e.f, args, e.type)

    # the MAIN CASE TO HANDLE in a Horner Scheme for Polynomials...
    elif etyp is NIR.Sum:
      # temporary: ignore the Horner Scheme benefits and just pass through
      #            for now
      terms = [ self.factorize(t) for t in e.terms ]
      return NIR.Sum( e.offset, terms, e.type )

    # Need to support factorization here, driven by
    # reasoning about the relative shapes of the factors
    elif etyp is NIR.Contract:
      n_factors = len(e.factors) + (1 if e.coeff != 1.0 else 0)

      if n_factors <= 2:
        factors = [ NIR.factor( self.factorize(f.expr), f.idx, f.power )
                    for f in e.factors ]
        return NIR.Contract(e.gen_binds, e.sum_binds,
                            e.coeff, e.preds, factors, e.type)

      # otherwise, we have some potential factorization work to do

      # defer to a System R style plan by exhaustive search strategy
      best_plan = self.SystemR_find_plan(e)

      # now execute the given factoring plan


    else: assert False, "Bad Case"


  def SystemR_find_plan(self, e):

    # We need some way to figure out which indices must be retained
    # for output, and which indices will be "equal" or nearly so
    # We get this data from the predicates...

    # This data will be abstracted into a list of tuples
    # each tuple will be a tuple of indices, indicating that there
    # is some sort of linear equation of those indices.
    #   e.g. i == 0  will result in a unary tuple
    #   e.g. i == j  will result in a binary tuple
    #   e.g. i == 3*j + k will result in a ternary tuple
    pred_rels = []
    for p in e.preds:
      assert type(p) is not NIR.Conj
      if type(p) is NIR.Alias:
        pred_rels.append( tuple(p.lvar.name, p.rvar.name) )
      elif type(p) is NIR.Cmp and p.op == '==':
        idx = [ t.ivar.name for t in p.eq.terms if t.ivar != None ]
        if len(idx) > 0:
          pred_rels.append( tuple(idx) )

    # We want to keep track of which variables have and haven't
    # "been summed" using a vector over the indices
    n_idx       = ( len(e.gen_binds) + len(e.sum_binds) +
                    sum(len(f.idx) for f in e.factors) )
    idx_summed  = [ False for _ in range(0,n_idx) ]
    # let's also keep track of the different index sets
    just_sum    = [ ib.ivar.name for ib in e.sum_binds ]
    just_gen    = [ ib.ivar.name for ib in e.gen_binds ]
    just_fac    = [ i for i in range(0,n_idx) if (i not in just_sum and
                                                  i not in just_gen) ]

    # we will also create a reference for looking up index sizes
    idx_size    = [ 1 for _ in range(0,n_idx) ]

    # to make our later work simpler, let's try to eliminate
    # the free summation indices now.
    for i in just_sum:
      i_summed  = False
      new_rels  = []
      for p in pred_rels:
        if i in p:
          # the first predicate we see gets removed
          # and fixes the value of i
          if not i_summed:
            # do not add to new_rels
            i_summed = True
            idx_summed[i] = True
          # additional predicates we see should be reduced to simulate
          # i now having an effectively constant value
          else:
            p = list(p)
            p.remove(i)
            if len(p) > 0:
              new_rels.append(p)
        # if this is some unaffected predicate, just copy it over
        else:
          new_rels.append(p)

      pred_rels = new_rels

    # save as a tuple...
    idx_summed = tuple(idx_summed)

    # A solution to cost-ing a sub-plan is encoded in a PlanLeaf(...) node
    #     - node  - is the solution
    #     - idx   - is a vector telling us which indices this
    #               sub-problem solution tensor is dimensioned over
    #     - summed- tells us which indices (globally) have been summed
    #               over in the process of solving this sub-problem
    #     - cost  - is an estimate of the work required to compute
    #               the sub-problem solution

    # here is how we lift the initial leaf nodes of a plan
    # into the sub-problem solution format.
    def coeff_leaf(c):
      return PlanLeaf(c,tuple(),tuple(),0)
    def factor_leaf(f):
      idx = [ False for _ in range(0,n_idx) ]
      for ib in f.idx:
        idx[ib.ivar.name] = True
      return PlanLeaf(f,tuple(idx),idx_summed,0)


    # now we can work out the cost of a plan
    def cost_plan(node):
      if type(node) is PlanLeaf:
        return node
      else:
        left      = cost_plan(node.left)
        right     = cost_plan(node.right)
        idx       = [ L or R for L,R in zip(left.idx, right.idx) ]
        summed    = [ L or R for L,R in zip(left.summed, right.summed) ]
        cost      = 1.0

        # whether an index is inside this node
        def node_i(i):
          return summed[i] or idx[i]
        # whether a pred-relation is entirely inside this node
        def node_p(p):
          return all( node_i(i) for i in p )
        # whether a pred-relation is already all summed
        def summed_p(p):
          return all( summed[i] for i in p )
        # whether a pred-relation spans this node and other indices
        def edge_p(p):
          c = sum( node_i(i) for i in p )
          return 0 < c < len(p)

        # first, we attempt to use predicate relations to
        # constrain the contraction further without having to
        # actually iterate over dimensions
        for p in pred_rels:
          if summed_p(p):
            pass # ignore such predicates
          elif node_p(p):
            # then find the largest unsummed index to sum over
            _,i = max([ (idx_size[i],i) for i in p if idx[i] ])
            # and move it from one list to the other
            idx[i] = False
            summed[i] = True
          # otherwise, we'll wait to process edge predicates...

        # now that we've eliminated as many indices as we can via
        # predicate constraints, we will split the remaining unsummed
        # indices into _output_ indices, ...
        out_idx   = [ False for _ in range(0,n_idx) ]
        for p in pred_rels:
          if edge_p(p):
            for i in p:
              if idx[i]:
                out_idx[i]  = True
        # ..., and _contracted_ indices
        for i,(is_out,is_idx) in enumerate(zip(out_idx,idx)):
          if is_idx and not is_out:
            cost     *= idx_size[i]
            summed[i] = True

        # we also need to cost iteration over all of the output
        # indices
        for i,out in enumerate(out_idx):
          if out:
            cost     *= idx_size[i]

        return PlanLeaf( PlanNode(left.node, right.node),
                         tuple(out_idx), tuple(summed),
                         cost + left.cost + right.cost )

    # given a costing function, we will enumerate all plans,
    # and pick the cheapest one

    # generate all partitions of a list into two lists
    # each of size 1 or greater
    # symmetry is removed by xs[0] always being in the left
    def gen_partitions(xs):
      # accomplished by enumerating all boolean vectors of the
      # form [True, x1, x2, ...] where xi may be either true or false
      for in_left in itertools.product([True,False], repeat = len(xs)-1 ):
        if all( b for b in in_left ):
          continue # must have at least one in the right
        in_left = (True,) + in_left
        left    = [ xs[i] for i,b in enumerate(in_left) if b ]
        right   = [ xs[i] for i,b in enumerate(in_left) if not b ]
        yield (left,right)

    def gen_plans(xs):
      assert len(xs) >= 1
      if len(xs) == 1:
        yield xs[0]
      elif len(xs) == 2:
        yield PlanNode(xs[0], xs[1])
      else:
        for L_xs, R_xs in gen_partitions(xs):
          for L_plan in gen_plans(L_xs):
            for R_plan in gen_plans(R_xs):
              yield PlanNode(L_plan, R_plan)

    # make the leaf nodes for plans
    leaves = [ factor_leaf(f) for f in e.factors ]
    if e.coeff != 1.0:
      leaves = [ coeff_leaf(e.coeff) ] + leaves

    # then generate the plans and find the minimum
    min_plan = PlanLeaf(None,None,None,math.inf)
    for plan in gen_plans(leaves):
      plan = cost_plan(plan)
      if plan.cost < min_plan.cost:
        min_plan = plan

    return min_plan





