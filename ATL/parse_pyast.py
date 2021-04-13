from __future__ import annotations


import inspect
import ast as pyast
import textwrap
from .prelude import *
from .quote_api import *
from . import builtins
from .frontend import UST,AST

import re

class Macro:
  def __init__(self,f,arg_names, global_vals,local_vals,getsrcinfo):
    self._macro_fdef  = f
    self._arg_names   = arg_names
    self._globals     = global_vals
    self._locals      = local_vals
    self._getsrcinfo  = getsrcinfo

  def expand(self,arg_quotes,):
    self._locals.push()
    # bind the arguments...
    if len(arg_quotes) != len(self._arg_names):
      raise TypeError(f"{self._getsrcinfo(self._macro_fdef)}: expected "
                      f"{len(self._arg_names)} arguments, but got "
                      f"{len(arg_quotes)}")
    for nm,q in zip(self._arg_names,arg_quotes):
      self._locals[nm] = q
    ust = parse_pyast( self._macro_fdef, self._globals, self._locals,
                                         self._getsrcinfo, as_macro = True )
    self._locals.pop()
    return ust

class Quote:
  def __init__(self,code, global_vals,local_vals,getsrcinfo):
    self._code  = code
    self._globals     = global_vals
    self._locals      = local_vals
    self._getsrcinfo  = getsrcinfo

  def parse(self,as_index):
    return parse_pyast( self._code, self._globals, self._locals,
                        self._getsrcinfo,
                        as_quote = True, as_index = as_index )

def macro(func):
  # note that we must dedent in case the function is defined
  # inside of a local scope
  rawsrc        = inspect.getsource(func)
  src           = textwrap.dedent(rawsrc)
  n_dedent      = ( len(re.match('^(.*)',rawsrc).group()) -
                    len(re.match('^(.*)',src).group()) )
  srcfilename   = inspect.getsourcefile(func)
  _, srclineno  = inspect.getsourcelines(func)
  srclineno    -= 1 # adjust for decorator line

  # convert into AST nodes; which should be a module with a single
  # FunctionDef node
  module    = pyast.parse(src)
  assert len(module.body) == 1
  assert type(module.body[0]) == pyast.FunctionDef

  # get global and local environments for context capture purposes
  func_globals  = func.__globals__
  stack_frames  = inspect.stack()
  assert(len(stack_frames) >= 1)
  assert(type(stack_frames[1]) == inspect.FrameInfo)
  func_locals   = stack_frames[1].frame.f_locals
  assert(type(func_locals) == dict)
  srclocals     = Environment(func_locals)
  for nm in ['sin','cos','sqrt','select_gt','ln','pow']:
    srclocals[nm]   = getattr(builtins, nm)
  srclocals['max']  = builtins.fmax
  srclocals['min']  = builtins.fmin

  def getsrcinfo(node):
    return SrcInfo( filename    = srcfilename,
                    lineno      = node.lineno+srclineno,
                    col_offset  = node.col_offset+n_dedent,
                    end_lineno  = (None if node.end_lineno is None
                                        else node.end_lineno+srclineno),
                    end_col_offset = (None if node.end_col_offset is None
                                        else node.end_col_offset+n_dedent) )

  def parse_macro_def(fdef):
    # certify simple argument form
    args    = fdef.args
    bad_arg_syntax_errmsg = """
    ATL expects macro arguments to not use the following python features:
      position-only arguments
      unnamed (position or keyword) arguments (i.e. *varargs, **kwargs)
      keyword-only arguments
      default argument values
    """
    if (len(args.posonlyargs) > 0 or args.vararg is not None or
        len(args.kwonlyargs) > 0 or len(args.kw_defaults) > 0 or
        args.kwarg is not None or len(args.defaults) > 0):
      raise TypeError(bad_arg_syntax_errmsg)

    argnames  = []
    for a in args.args:
      if a.annotation is not None:
        raise TypeError(f"{getsrcinfo(a)}: expected untyped macro args")
      argnames.append(a.arg)

    if fdef.returns is not None:
      raise TypeError(f"{getsrcinfo(fdef)}: macros should not have "
                      f"return type annotations")

    return Macro(fdef, argnames, func_globals, srclocals, getsrcinfo)

  return parse_macro_def(module.body[0])

def func(func):
  # note that we must dedent in case the function is defined
  # inside of a local scope
  rawsrc        = inspect.getsource(func)
  src           = textwrap.dedent(rawsrc)
  n_dedent      = ( len(re.match('^(.*)',rawsrc).group()) -
                    len(re.match('^(.*)',src).group()) )
  srcfilename   = inspect.getsourcefile(func)
  _, srclineno  = inspect.getsourcelines(func)
  srclineno    -= 1 # adjust for decorator line

  # convert into AST nodes; which should be a module with a single
  # FunctionDef node
  module    = pyast.parse(src)
  assert len(module.body) == 1
  assert type(module.body[0]) == pyast.FunctionDef

  # get global and local environments for context capture purposes
  func_globals  = func.__globals__
  stack_frames  = inspect.stack()
  assert(len(stack_frames) >= 1)
  assert(type(stack_frames[1]) == inspect.FrameInfo)
  func_locals   = stack_frames[1].frame.f_locals
  assert(type(func_locals) == dict)
  srclocals     = Environment(func_locals)
  for nm in ['sin','cos','sqrt','select_gt','ln','pow']:
    srclocals[nm]   = getattr(builtins, nm)
  srclocals['max']  = builtins.fmax
  srclocals['min']  = builtins.fmin

  def getsrcinfo(node):
    return SrcInfo( filename    = srcfilename,
                    lineno      = node.lineno+srclineno,
                    col_offset  = node.col_offset+n_dedent,
                    end_lineno  = (None if node.end_lineno is None
                                        else node.end_lineno+srclineno),
                    end_col_offset = (None if node.end_col_offset is None
                                        else node.end_col_offset+n_dedent) )

  parsed_ust = parse_pyast( module.body[0], func_globals, srclocals,
                                            getsrcinfo, as_func = True )

  return Function(parsed_ust)


def parse_pyast( module_ast, func_globals, srclocals, getsrcinfo,
                 as_func = False, as_macro = False,
                 as_quote = False, as_index = False ):

  # define evaluation of type annotations
  def eval_expr(expr):
    assert isinstance(expr, pyast.expr)
    code  = compile(pyast.Expression(expr),'','eval')
    e_obj = eval(code, func_globals, srclocals)
    return e_obj
  def eval_type_annotation(expr):
    return Type(eval_expr(expr))
  def eval_range_annotation(expr):
    e_obj = eval_expr(expr)
    if is_pos_int(e_obj):
      return e_obj
    elif type(e_obj) is Size:
      return e_obj._name
    else:
      raise TypeError(f"{getsrcinfo(expr)}: expected a range (size or int)")

  # ******
  # Start the definition of parsing routines...
  def parse_fdef(fdef):
    # certify simple argument form
    args    = fdef.args
    bad_arg_syntax_errmsg = """
    ATL expects function arguments to not use the following python features:
      position-only arguments
      unnamed (position or keyword) arguments (i.e. *varargs, **kwargs)
      keyword-only arguments
      default argument values
    """
    if (len(args.posonlyargs) > 0 or args.vararg is not None or
        len(args.kwonlyargs) > 0 or len(args.kw_defaults) > 0 or
        args.kwarg is not None or len(args.defaults) > 0):
      error(bad_arg_syntax_errmsg)

    # process each argument in order
    arg_order   = []
    var_decls   = []
    size_decls  = []
    rel_decls   = []
    #print(args.args)
    for a in args.args:
      #print(a.arg, a.annotation)
      assert a.annotation is not None
      tnode = a.annotation
      arg_order.append(a.arg)
      if type(tnode) is pyast.Name and tnode.id == 'size':
        sz                  = Size(a.arg)
        sz._srcinfo         = getsrcinfo(a)
        srclocals[a.arg]    = sz
        size_decls.append(UST.size_decl(sz._name, sz._srcinfo))
      elif type(tnode) is pyast.Tuple: # detected relation argument
        srclocals[a.arg]    = Relation(a.arg)
        srclocals[a.arg]._srcinfo   = getsrcinfo(a)
        dims  = [ eval_range_annotation(e) for e in tnode.elts ]
        rel_decls.append(UST.rel_decl( a.arg, dims, getsrcinfo(a) ))
      else:
        typ = eval_type_annotation(tnode)._type
        srclocals[a.arg]    = a.arg
        var_decls.append(UST.var_decl( a.arg, typ, getsrcinfo(a) ))
    rtyp    = None
    if fdef.returns is not None:
      rtyp  = eval_type_annotation(fdef.returns)._type

    srclocals.push()
    body    = parse_stmt_block(fdef)
    srclocals.pop()
    return UST.function( name         = fdef.name,
                         arg_order    = arg_order,
                         rettype      = rtyp,
                         vars         = var_decls,
                         sizes        = size_decls,
                         relations    = rel_decls,
                         body         = body,
                         srcinfo      = getsrcinfo(fdef) )

  def parse_stmt_block(fdef):
    """parse a function body"""
    body    = fdef.body
    stmts   = []
    for s in body[0:-1]:
      if type(s) is pyast.Return:
        raise TypeError(f"{getsrcinfo(s)}: return may only occur "
                        f"at end of function")
      elif type(s) is pyast.Assign:
        if len(s.targets) != 1:
          raise TypeError(f"{getsrcinfo(s)}: "
                          f"expected single left-hand side of assignments")
        lhs = s.targets[0]
        rhs = s.value
        stmts.append( parse_asgn(lhs,rhs,s) )

      elif type(s) is pyast.AnnAssign:
        raise TypeError(f"{getsrcinfo(s)}: "
                        f"assignment annotations unimplemented")
      elif type(s) is pyast.AugAssign:
        raise TypeError(f"{getsrcinfo(s)}: "
                        f"reduction statements unimplemented")
      else:
        raise TypeError(f"{getsrcinfo(s)}: unexpected statement type '{styp}'")
    if type(body[-1]) is not pyast.Return:
      raise TypeError(f"{getsrcinfo(s)}: "
                      f"expected last statement to be a return")
    ret_expr  = parse_expr(body[-1].value)
    # extract values and package into UST object directly...
    return UST.Let(stmts, ret_expr, getsrcinfo(fdef))

  def parse_slices_as_ranges(slices):
    if type(slices) is pyast.ExtSlice:
      ss      = slices.dims
    else:
      ss      = [ slices ]

    idxs    = []
    for s in ss:
      if type(s) is pyast.Index:
        raise TypeError(f"{getsrcinfo(s)} expected lhs index variables "
                        f"to be declared in the form 'name : range'")
      assert type(s) is pyast.Slice
      if s.step is not None:
        raise TypeError(f"{getsrcinfo(s)} expected 'i : n', not 'i : n : m'")
      if s.lower is None or s.upper is None:
        raise TypeError(f"{getsrcinfo(s)} expected 'i:n'")

      if type(s.lower) is not pyast.Name:
        raise TypeError(f"{getsrcinfo(s)} expected simple name 'i' in 'i:n'")
      
      rng   = eval_range_annotation(s.upper)
      ivar  = s.lower.id
      idxs.append( (ivar,rng) )

    return idxs

  def parse_asgn(lhs,rhs,assgn):
    """parse an assignment"""
    srclocals.push()
    varname   = None
    idxs      = []
    etyp      = type(lhs)
    if etyp is pyast.Name:
      varname = lhs.id
    elif etyp is pyast.Subscript:
      if type(lhs.value) is not pyast.Name:
        raise TypeError(f"{getsrcinfo(e)}: expected name[...] form on "
                        f"left-hand-side of assignment")
      varname = lhs.value.id
      # parse out slices...
      idxs = parse_slices_as_ranges(lhs.slice)
      for nm,_ in idxs:
        srclocals[nm] = nm
    else:
      raise TypeError(f"{getsrcinfo(lhs)}: unexpected expression "
                      f"on lhs '{etyp}'")

    # now parse the right-hand-side
    r_expr    = parse_expr(rhs)
    for nm,rng in reversed(idxs):
      r_expr  = UST.Gen(nm,rng,r_expr,getsrcinfo(lhs))
    srclocals.pop()

    srclocals[varname] = varname
    vartyp    = None
    return UST.assign(varname, vartyp, r_expr, getsrcinfo(assgn))

  def parse_expr(e, expect_index=False):
    etyp    = type(e)
    if etyp is pyast.BoolOp:
      vals  = [ parse_expr(v) for v in e.values ]
      p     = vals[0]
      for rhs in vals[1:]:
        if type(e.op) is pyast.And:
          p = UST.Conj(p, rhs, getsrcinfo(e))
        elif type(e.op) is pyast.Or:
          p = UST.Disj(p, rhs, getsrcinfo(e))
        else: assert False, "impossible branch"
      return p
    elif etyp is pyast.BinOp:
      if expect_index:
        lhs     = parse_expr(e.left,expect_index=True)
        rhs     = parse_expr(e.right,expect_index=True)
        if type(lhs) is UST.Const and type(rhs) is UST.Const:
          if   type(e.op) == pyast.Add:   v = lhs.val + rhs.val
          elif type(e.op) == pyast.Sub:   v = lhs.val - rhs.val
          elif type(e.op) == pyast.Mult:  v = lhs.val * rhs.val
          else:
            raise TypeError(f"{getsrcinfo(e)}: unsupported operator: {e.op}")
          return UST.Const(v,getsrcinfo(e))
        elif type(e.op) == pyast.Mult:
          if type(lhs) is UST.Const:
            return UST.IdxScale(Fraction(lhs.val),rhs,getsrcinfo(e))
          elif type(rhs) is UST.Const:
            return UST.IdxScale(Fraction(rhs.val),lhs,getsrcinfo(e))
          else:
            raise TypeError(f"{getsrcinfo(e)}: cannot multiply two indices; "
                            f"doing so would create non-linear indexing")
        elif type(e.op) == pyast.Add:
          return UST.IdxAdd(lhs, rhs, getsrcinfo(e))
        elif type(e.op) == pyast.Sub:
          rhs_ast   = UST.IdxScale(Fraction(-1), rhs, getsrcinfo(e))
          return UST.IdxAdd(lhs, rhs_ast, getsrcinfo(e))
        else:
          raise TypeError(f"{getsrcinfo(e)}: unsupported operator: {e.op}")
      else:
        lhs     = parse_expr(e.left)
        rhs     = parse_expr(e.right)
        if   type(e.op) == pyast.Add:     op = '+'
        elif type(e.op) == pyast.Sub:     op = '-'
        elif type(e.op) == pyast.Mult:    op = '*'
        elif type(e.op) == pyast.Div:     op = '/'
        else:
          raise TypeError(f"{getsrcinfo(e)}: unsupported operator: {e.op}")

        if op == '*':
          if isinstance(lhs, UST.pred) and isinstance(rhs, UST.pred):
            raise TypeError(f"{getsrcinfo(e)}: cannot multiply two predicates")
          elif isinstance(lhs, UST.pred):
            return UST.Indicate(lhs, rhs, getsrcinfo(e))
          elif isinstance(rhs, UST.pred):
            return UST.Indicate(rhs, lhs, getsrcinfo(e))
        # finally, if op isn't '*' or lhs and rhs are both expressions
        if isinstance(lhs, UST.pred):
          raise TypeError(f"{lhs.srcinfo}: expected expression, not predicate")
        elif isinstance(rhs, UST.pred):
          raise TypeError(f"{rhs.srcinfo}: expected expression, not predicate")
        return UST.BinOp( op, lhs, rhs, getsrcinfo(e) )
    elif etyp is pyast.UnaryOp:
      # hack for negated constants
      if type(e.op) is pyast.USub:
        if type(e.operand) is pyast.Constant:
          c = parse_expr(e.operand, expect_index=expect_index)
          c.val = -c.val
          return c
      raise TypeError(f"{getsrcinfo(e)}: unary operations unimplemented")
    elif etyp is pyast.Compare:
      args  = [ parse_expr(v,expect_index=True) for v in e.comparators ]
      lhs   = parse_expr(e.left,expect_index=True)
      conj  = None
      for op,rhs in zip(e.ops,args):
        if   type(op) == pyast.Eq:      opstr = '=='
        elif type(op) == pyast.Lt:      opstr = '<'
        elif type(op) == pyast.Gt:      opstr = '>'
        elif type(op) == pyast.LtE:     opstr = '<='
        elif type(op) == pyast.GtE:     opstr = '>='
        elif type(op) == pyast.NotEq:
          raise TypeError(f"{getsrcinfo(op)}: unsupported comparison: !=")
        else:
          raise TypeError(f"{getsrcinfo(op)}: unsupported comparison: {op}")
        p   = UST.Cmp(opstr, lhs, rhs, getsrcinfo(e))
        if conj is None:
          conj  = p
        else:
          conj  = UST.Conj(conj, p, getsrcinfo(e))
      return conj
    elif etyp is pyast.Call:
      if len(e.keywords) > 0:
        raise TypeError(f"{getsrcinfo(e)}: keyword arguments unsupported")

      # is this a big op?
      if type(e.func) is pyast.Subscript:
        # e.func.value, e.func.slice
        if (type(e.func.value) is pyast.Name and
            (e.func.value.id == "Sum" or e.func.value.id == "Gen")):
          if len(e.args) != 1:
            raise TypeError(f"{getsrcinfo(e)}: expected exactly one argument")
          bigop   = UST.Sum if e.func.value.id == "Sum" else UST.Gen

          srclocals.push()
          idxs    = parse_slices_as_ranges(e.func.slice)
          for nm,_ in idxs:
            srclocals[nm] = nm
          body    = parse_expr(e.args[0])
          for nm,rng in reversed(idxs):
            body  = bigop(nm,rng,body,getsrcinfo(e.func))
          srclocals.pop()

          return body
        else:
          raise TypeError(f"{getsrcinfo(e.func.value)}: "
                          f"expected 'Sum' or 'Gen'")

      # otherwise this is a normal call, so
      # we want to evaluate the left-hand-side to a function object
      # or to a built-in math function, or to a relation argument
      else:
        func    = eval_expr(e.func)
        if type(func) is Function:
          # process args specially to extract size and relation variables
          ast   = func._ast
          decls = ast.decls_in_order()
          args  = []
          if len(decls) != len(e.args):
            raise TypeError(f"{getsrcinfo(e)}: expected {len(decls)} "
                            f"arguments to function '{ast.name}', but "
                            f"only got {len(e.args)} arguments.")
          for d,a in zip(decls,e.args):
            if type(d) is AST.size_decl:
              rng = eval_range_annotation(a)
              args.append(rng)
            elif type(d) is AST.rel_decl:
              if type(a) is not pyast.Name:
                raise TypeError(f"{getsrcinfo(a)}: relation argument "
                                f"expected a simple name")
              rel = eval_expr(a)
              if type(rel) is not Relation:
                raise TypeError(f"{getsrcinfo(a)}: expected '{a.id}' to "
                                f" be a relation")
              args.append(rel._name)
            else:
              assert type(d) is AST.var_decl
              args.append(parse_expr(a))
          # then package up as a func call
          return UST.FuncCall(ast, args, getsrcinfo(e))
        elif type(func) is Macro:
          return func.expand([ Quote(code,func_globals,srclocals,getsrcinfo)
                               for code in e.args ])
        elif isinstance(func, builtins.BuiltIn):
          args  = [ parse_expr(a) for a in e.args ]
          return UST.BuiltIn(func, args, getsrcinfo(e))
        elif type(func) is Relation:
          iargs = [ parse_expr(i,expect_index=True) for i in e.args ]
          return UST.Relation(func._name, iargs, getsrcinfo(e))
        else:
          raise TypeError(f"{getsrcinfo(e.func)}: expected a function or "
                          f"builtin")
    elif etyp is pyast.Constant:
      if expect_index:
        if type(e.value) is not int:
          raise TypeError(f"{getsrcinfo(e)}: expected int")
        return UST.IdxConst(e.value, getsrcinfo(e))
      else:
        if type(e.value) is not float and type(e.value) is not int:
          raise TypeError(f"{getsrcinfo(e)}: expected float or int")
        return UST.Const(float(e.value), getsrcinfo(e))
    elif etyp is pyast.Attribute:
      arg     = parse_expr(e.value)
      if e.attr[0] == '_' and e.attr[1:].isdigit():
        idx = int(e.attr[1:])
      else:
        idx = e.attr
      return UST.Proj(idx,arg,getsrcinfo(e))
    elif etyp is pyast.Subscript:
      base      = parse_expr(e.value)

      if type(e.slice) is not pyast.Index:
        raise TypeError(f"{getsrcinfo(s)} expected indexing without ranges "
                        f"i.e. as 'i', not 'i:j'")
      elif type(e.slice.value) is pyast.Tuple:
        elts    = e.slice.value.elts
      else:
        elts    = [ e.slice.value ]

      idxs      = []
      for sube in elts:
        idxs.append( parse_expr(sube,expect_index=True) )
      return UST.Access( base, idxs, getsrcinfo(e) )
    elif etyp is pyast.Name:
      lookup  = eval_expr(e)
      if expect_index:
        if type(lookup) is Size:
          return UST.IdxSize(lookup._name, getsrcinfo(e))
        elif type(lookup) is int:
          return UST.IdxConst(lookup, getsrcinfo(e))
        elif type(lookup) is str:
          return UST.IdxVar(lookup, getsrcinfo(e))
        elif type(lookup) is Quote:
          return lookup.parse(as_index=True)
        else:
          raise TypeError(f"{getsrcinfo(e)}: name '{e.id}' had "
                          f"unexpected type {type(lookup)}")
      else:
        if type(lookup) is float or type(lookup) is int:
          return UST.Const(float(lookup), getsrcinfo(e))
        elif type(lookup) is str:
          return UST.Var(lookup, getsrcinfo(e))
        elif type(lookup) is Relation: # needed for function calls
          return lookup._name # str
        elif type(lookup) is Size: # needed for function calls
          return lookup._name # Sym
        elif type(lookup) is Quote:
          return lookup.parse(as_index=False)
        else:
          raise TypeError(f"{getsrcinfo(e)}: name '{e.id}' had "
                          f"unexpected type {type(lookup)}")
    elif etyp is pyast.List:
      if len(e.elts) < 1:
        raise TypeError(f"{getsrcinfo(e)}: tensor expressions must have at "
                        f"least one entry")
      elts    = [ parse_expr(te) for te in e.elts ]
      return UST.TensorLit( elts, getsrcinfo(e) )
    elif etyp is pyast.Tuple:
      if len(e.elts) == 1:
        return parse_expr(e.elts[0],expect_index=expect_index)
      elif len(e.elts) < 1:
        raise TypeError(f"{getsrcinfo(e)}: tuple expressions must have at "
                        f"least one entry")
      elts    = [ parse_expr(te) for te in e.elts ]
      return UST.Tuple( elts, getsrcinfo(e) )
    elif etyp is pyast.Dict:
      assert len(e.keys) == len(e.values)
      if len(e.values) < 1:
        raise TypeError(f"{getsrcinfo(e)}: tuple/record expressions "
                        f"must have at least one entry")
      vals    = [ parse_expr(te) for te in e.values ]
      labels  = []
      for ke in e.keys:
        if type(ke) is not pyast.Name:
          raise TypeError(f"{getsrcinfo(e)}: expected named-tuples/records "
                          f"to have simple labels")
        labels.append(ke.id)
      return UST.NamedTuple( labels, vals, getsrcinfo(e) )

    else:
      raise TypeError(f"{getsrcinfo(e)}: unexpected expression type '{etyp}'")

    #if expect_index:
    #  return UST.IdxConst(0, getsrcinfo(e))
    #else:
    #  return UST.Const( 0.0, getsrcinfo(e) )

  if as_func:
    return parse_fdef(module_ast)
  elif as_macro:
    return parse_stmt_block(module_ast)
  elif as_quote:
    return parse_expr(module_ast, expect_index=as_index)
  else:
    error("unexpected case; need to specify what to parse as")












