
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from . import builtins as B 

from fractions import Fraction

from .frontend import AST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# An IR that is a partial normalization of the ATL ASTs

norm_pred_ops = {
  ">"   : True,
  ">="  : True,
  "=="  : True,
}

# Typed AST
NIR = ADT("""
module NIR {
  function  = ( name?       name,
                sym*        arg_order,
                type        rettype,
                var_decl*   vars,
                size_decl*  sizes,
                rel_decl*   relations,
                tuple_in*   tuple_vars,
                tuple_out*  outputs,
                srcinfo     srcinfo )

  var_decl  = ( sym name, type   type,  srcinfo srcinfo )
  size_decl = ( sym name,               srcinfo srcinfo )
  rel_decl  = ( sym name, range* sizes, srcinfo srcinfo )
  tuple_in  = ( sym name, plabel* proj )
  tuple_out = Tuple( tuple_out* args )
            | OutLeaf( expr expr )
            attributes( type type, srcinfo srcinfo )

  expr  = Var       ( sym       name    )
        | Const     ( float     val     )
        | Zero      ()
        | Sum       ( float     offset,
                      expr*     terms   )
        | Contract  ( idx_bind* gen_names,
                      idx_bind* sum_names,
                      float     coeff, 
                      pred*     preds,
                      factor*   factors )
        | Exp       ( expr base, fraction power )
        | BuiltIn   ( builtin f, expr* args )
        attributes( type type )

  idx_bind  = ( sym name, range range )
  factor    = ( expr base, sym* idx )

  affine    = ( fraction offset, aterm* terms )
  aterm     = ( fraction coeff,  sym    name,   bool is_size )
  
  pred  = Cmp       ( pred_op op, affine eq ) -- e.g.  eq > 0
        | Relation  ( sym name,   sym* args )
        | Conj      ( pred* preds )
        | Disj      ( pred* preds )
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'builtin':  lambda x: isinstance(x, B.BuiltIn),
  'pred_op':  lambda x: x in norm_pred_ops,
  'plabel':   lambda x: type(x) is int,
  'srcinfo':  lambda x: type(x) is SrcInfo,
})

ADTmemo(NIR,[
  'Var', 'Const', 'Zero', 'Sum', 'Contract', 'Exp', 'BuiltIn',
  'idx_bind', 'factor', 'affine', 'aterm',
  'Cmp', 'Relation', 'Conj', 'Disj',
],{
  'name':     lambda x: x,
  'sym':      lambda x: x,
  'type':     id,
  'range':    lambda x: x,
  'fraction': lambda x: x,
  'builtin':  id,
  'pred_op':  lambda x: x,
  'plabel':   lambda x: x,
})

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class AST_to_NIR:
  def __init__(self, f):
    self._ctxt    = Context()

    # setup initial stuff somehow...


  def nir(self):
    return self._nir

  def normalize(self, e):
    eclass  = type(e)
    if   eclass is AST.Var:
      # lookup and return

    elif eclass is AST.Const:
      


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #



