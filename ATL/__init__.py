
""" ATL - A Tensor Library """

from .quote_api import (
  Function,
  Var,
  IVar,
  Relation,
  Size,
  Type,
  Sum,
  Gen,
  Let,
  Tuple,
  Expr,
  IExpr,
  Pred,
  Fun,
  ATLmath,
)
num = Type(float)

from .atl_types import is_type,  Num, Tuple, Tensor

from .parse_pyast import func, macro

__all__ = [
  "Function",
  #
  "Var",
  "IVar",
  "Relation",
  "Size",
  "Type",
  "Sum",
  "Gen",
  "Let",
  "Tuple",
  "Expr",
  "IExpr",
  "Pred",
  "Fun",
  #
  "ATLmath",
  #
  "num",
  #
  "func",
  "macro",
  #
  "is_type",
  "Num",
  "Tuple",
  "Tensor",
]