
from .adt import ADT
from .adt import memo as ADTmemo

from .prelude import *

from . import atl_types as T
from . import builtins as B 

from fractions import Fraction

from .frontend import pred_ops, AST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# An IR that is a partial normalization of the ATL ASTs
# this form 



# Typed AST
NIR = ADT("""
module NIR {
  function  = ( name?       name,
                sym*        arg_order,
                type        rettype,
                var_decl*   vars,
                size_decl*  sizes,
                rel_decl*   relations,
                expr        body,
                srcinfo     srcinfo )

  var_decl  = ( sym name, type   type,  srcinfo srcinfo )
  size_decl = ( sym name,               srcinfo srcinfo )
  rel_decl  = ( sym name, range* sizes, srcinfo srcinfo )




  expr = Var      ( sym   name )
        --| Const    ( float val  )
       | Add      ( float offset, expr* terms   )
       | Mul      ( float coeff,  expr* factors )
       | Exp      ( expr base, int power )
        --| Add      ( expr lhs, expr rhs )
        --| Mul      ( expr lhs, expr rhs )
       | Tuple    ( expr* args )
       | Proj     ( plabel idx, expr arg )
       | TensorLit( expr* args )
       | Gen      ( sym name, range range, expr body )
       | Sum      ( sym name, range range, expr body )
       | Access   ( expr base, index* idx )
       | BuiltIn  ( builtin f, expr* args )
       -- implied multiplication of the bracket with body
       | Indicate ( pred pred, expr body  )
       -- important to express sharing of computation results
       | Let      ( assign* stmts, expr ret )
       attributes( type type, srcinfo srcinfo )

  assign  = ( sym name, type type, expr rhs, srcinfo srcinfo )
  
  index   = IdxConst  ( int      val  )
          | IdxVar    ( sym      name )
          | IdxSize   ( sym      name )
          | IdxAdd    ( index    lhs,   index rhs )
          | IdxScale  ( fraction coeff, index idx )
          attributes( srcinfo srcinfo )
  
  pred    = Cmp       ( pred_op op, index lhs,  index rhs )
          | Relation  ( sym name,   index* args )
          | Conj      ( pred lhs,   pred rhs    )
          | Disj      ( pred lhs,   pred rhs    )
          attributes( srcinfo srcinfo )
}
""", {
  'name':     is_valid_name,
  'sym':      lambda x: type(x) is Sym,
  'type':     T.is_type,
  'range':    lambda x: is_pos_int(x) or (type(x) is Sym),
  'fraction': lambda x: type(x) is Fraction,
  'builtin':  lambda x: isinstance(x, B.BuiltIn),
  'pred_op':  lambda x: x in pred_ops,
  'plabel':   lambda x: x is int,
  'srcinfo':  lambda x: type(x) is SrcInfo,
})









