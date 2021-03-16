
import unittest
from ATL import *

import numpy as np
import random
import time

from functools import wraps
from collections import namedtuple

from re import compile as _re_compile
_data_method_RE = _re_compile("^data_")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

DataCase    = namedtuple('DataCase',['name','input','output'])
InputCase   = namedtuple('InputCase',['name','input'])
DInputCase  = namedtuple('DInputCase',['name','input','dinput'])
DInOutCase  = namedtuple('DInOutCase',['name','input','dinput','doutput'])

class FunctionTestCase:
  """ A testing mixin that should be used alongside unittest.TestCase
  Expects gen_func() to be defined returning an ATL function.  Then every
  method with name data_*() is expected to return a sample input/output
  data pair for testing the function with.
  """

  def _init_rand(self):
    if not hasattr(self,'rand'):
      self.rand = RandKey()

  def gen_func(self):
    raise NotImplementedError()

  def gen_func_memo(self):
    if not hasattr(self, '_ftc_gen_func_memoized'):
      self._ftc_gen_func_memoized = self.gen_func()
    return self._ftc_gen_func_memoized

  def gen_deriv(self):
    raise NotImplementedError()

  def gen_deriv_memo(self):
    if not hasattr(self, '_ftc_gen_deriv_memoized'):
      self._ftc_gen_deriv_memoized = self.gen_deriv()
    return self._ftc_gen_deriv_memoized

  def gen_deriv_sig(self):
    raise NotImplementedError()

  def rand_input(self):
    raise NotImplementedError()

  def rand_deriv_inout(self):
    raise NotImplementedError()

  def rand_deriv_output(self):
    raise NotImplementedError()

  def rand_perf_inout(self):
    raise NotImplementedError()

  def n_rand_tests(self):
    return 2

  # --------------------------- #

  def discover_data(self):
    if not hasattr(self, '_ftc_inout_data'):
      self._init_rand()

      data_fstrs  = [ f for f in dir(self)
                        if _data_method_RE.match(f) != None ]
      data_funcs  = [ (f, getattr(self, f)) for f in data_fstrs ]
      data_inout  = []
      for fname,gen_data in data_funcs:
        i,o       = gen_data()
        data_inout.append( DataCase(fname,i,o) )
      self._ftc_inout_data = data_inout

    return self._ftc_inout_data

  def discover_rand_data(self):
    if not hasattr(self, '_ftc_rand_data'):
      self._init_rand()
      data_cases  = self.discover_data().copy()
      n_default   = self.n_rand_tests() - len(data_cases)
      func        = self.gen_func_memo()

      for i in range(0,n_default):
        in_data   = self.rand_input()
        out_data  = func.interpret(*in_data)
        data_cases.append( DataCase(f"rand_{i}",in_data,out_data) )

      self._ftc_rand_data = data_cases

    return self._ftc_rand_data

  def discover_rand_datum(self):
    if not hasattr(self, '_ftc_rand_datum'):
      self._init_rand()
      func        = self.gen_func_memo()
      in_data     = self.rand_input()
      out_data    = func.interpret(*in_data)
      self._ftc_rand_datum = DataCase(f"rand_datum",in_data,out_data)

    return self._ftc_rand_datum

  def discover_rand_deriv_datum(self):
    if not hasattr(self, '_ftc_rand_deriv_datum'):
      self._init_rand()
      in_data, d_in_data          = self.rand_deriv_input()
      self._ftc_rand_deriv_datum  = DInputCase(f"rand_deriv_datum",
                                               in_data,d_in_data)

    return self._ftc_rand_deriv_datum

  def discover_rand_deriv_input(self):
    if not hasattr(self, '_ftc_rand_input_data'):
      self._init_rand()

      data_cases  = []
      n_default   = self.n_rand_tests()

      for i in range(0,n_default):
        in_data, d_in_data      = self.rand_deriv_input()
        data_cases.append( DInputCase(f"rand_deriv_{i}",in_data,d_in_data) )

      self._ftc_rand_input_data = data_cases

    return self._ftc_rand_input_data

  def discover_rand_adjoint_inout(self):
    if not hasattr(self, '_ftc_rand_inout_data'):
      self._init_rand()

      data_cases  = []
      n_default   = self.n_rand_tests()

      for i in range(0,n_default):
        in_data,din,dout        = self.rand_deriv_inout()
        data_cases.append( DInOutCase(f"rand_adjoint_{i}",in_data,din,dout) )

      self._ftc_rand_inout_data = data_cases

    return self._ftc_rand_inout_data

  def discover_rand_perf_inout(self):
    if not hasattr(self, '_ftc_rand_perf_inout_data'):
      self._init_rand()

      data_cases  = []
      n_default   = self.n_rand_tests()

      for i in range(0,n_default):
        in_data,din,dout        = self.rand_perf_inout()
        data_cases.append( DInOutCase(f"rand_perf_{i}",in_data,din,dout) )

      self._ftc_rand_perf_inout_data = data_cases

    return self._ftc_rand_perf_inout_data


  # --------------------------- #
  
  def test_print(self):
    self.gen_func_memo()

  def test_print_adjoint(self):
    deriv_sig     = self.gen_deriv_sig()
    dfunc         = self.gen_func_memo()._TEST_NIR_Adjoint(**deriv_sig)
    #print(dfunc)

  def test_print_nir_roundtrip(self):
    nir_func      = self.gen_func_memo()._TEST_NIR_Roundtrip_YesSimp()
    #print(nir_func)

  def test_interpreter(self):
    for fname,in_data,out_data in self.discover_data():
      with self.subTest(data=fname):
        func              = self.gen_func_memo()
        comp_out          = func.interpret(*in_data)
        np.testing.assert_allclose(comp_out, out_data)

  def test_compiler(self):
    func                  = self.gen_func_memo()
    func                  = func._TEST_PreNormalization()
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        #interp_out        = func.jit_exec(*in_data)
        interp_out        = func.cjit(*in_data)
        print(in_data)
        print(func._cjit_compiled.codestr())
        print(out_data, interp_out)
        np.testing.assert_allclose(out_data, interp_out)

  def test_let_lift(self):
    func                  = self.gen_func_memo()
    liftfunc              = func._TEST_LetLift()
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        lift_out          = liftfunc.interpret(*in_data)
        np.testing.assert_allclose(out_data, lift_out)

  def test_tuple_elim(self):
    func                  = self.gen_func_memo()
    liftfunc              = func._TEST_TupleElimination()
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        lift_out          = liftfunc.interpret(*in_data)
        np.testing.assert_allclose(out_data, lift_out)

  def test_pre_normalization(self):
    func                  = self.gen_func_memo()
    liftfunc              = func._TEST_PreNormalization()
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        lift_out          = liftfunc.interpret(*in_data)
        np.testing.assert_allclose(out_data, lift_out)

  def test_nir_roundtrip_no_simp(self):
    # only test with one datum pair because we expect it's inefficient
    fname,in_data,out_data = self.discover_rand_datum()
    with self.subTest(data=fname):
      func                = self.gen_func_memo()
      nir_func            = func._TEST_NIR_Roundtrip_NoSimp()
      nir_out             = nir_func.interpret(*in_data)
      np.testing.assert_allclose(out_data, nir_out)

  def test_nir_roundtrip_yes_simp(self):
    # TODO: change back to multiple data
    fname,in_data,out_data = self.discover_rand_datum()
    with self.subTest(data=fname):
      func                = self.gen_func_memo()
      nir_func            = func._TEST_NIR_Roundtrip_YesSimp()
      nir_out             = nir_func.interpret(*in_data)
      np.testing.assert_allclose(out_data, nir_out)

  def test_filterdown(self):
    func                  = self.gen_func_memo()
    nir_func              = func._TEST_NIR_Roundtrip_YesSimp()
    nir_func              = nir_func._TEST_NIR_filterdown()
    #print(nir_func)
    fname,in_data,out_data = self.discover_rand_datum()
    with self.subTest(data=fname):
      nir_out             = nir_func.interpret(*in_data)
      np.testing.assert_allclose(out_data, nir_out)

  def test_total_derivative_alone(self):
    func                  = self.gen_func_memo()
    deriv_sig             = self.gen_deriv_sig()
    dfunc                 = func._TEST_TotalDeriv_Alone(**deriv_sig)
    ref_dfunc             = self.gen_deriv_memo()
    for fname,in_data,d_in_data in self.discover_rand_deriv_input():
      with self.subTest(data=fname):
        df_out            = dfunc.interpret(*(in_data+d_in_data))
        ref_out           = ref_dfunc.interpret(*(in_data+d_in_data))
        np.testing.assert_allclose(df_out, ref_out)

  def test_derivative(self):
    func                  = self.gen_func_memo()
    deriv_sig             = self.gen_deriv_sig()
    dfunc                 = func._TEST_TotalDeriv(**deriv_sig)
    ref_dfunc             = self.gen_deriv_memo()
    for fname,in_data,d_in_data in self.discover_rand_deriv_input():
      with self.subTest(data=fname):
        df_out            = dfunc(*(in_data+d_in_data))
        ref_out           = ref_dfunc(*(in_data+d_in_data))
        np.testing.assert_allclose(df_out, ref_out)

  def test_nir_derivative(self):
    func                  = self.gen_func_memo()
    deriv_sig             = self.gen_deriv_sig()
    dfunc                 = func._TEST_NIR_Deriv(**deriv_sig)
    ref_dfunc             = self.gen_deriv_memo()
    for fname,in_data,d_in_data in self.discover_rand_deriv_input():
      with self.subTest(data=fname):
        df_out            = dfunc(*(in_data+d_in_data))
        ref_out           = ref_dfunc(*(in_data+d_in_data))
        np.testing.assert_allclose(df_out, ref_out)

  def test_nir_adjoint(self):
    func                  = self.gen_func_memo()
    deriv_sig             = self.gen_deriv_sig()
    d_in_is_tup           = (len(deriv_sig) > 1)
    adj_func              = func._TEST_NIR_Adjoint(**deriv_sig)
    ref_dfunc             = self.gen_deriv_memo()
    for fname,in_data,d_in,d_out in self.discover_rand_adjoint_inout():
      with self.subTest(data=fname):
        #print('D_IN\n',d_in)
        #print('D_OUT\n',d_out)
        adj_out           = adj_func(*( in_data+(d_out,) ))
        ref_out           = ref_dfunc(*(in_data+d_in))
        adj_grad_out      = ( adj_out.grad_out if d_in_is_tup
                                             else (adj_out.grad_out,) )
        #print('adj_out\n',adj_out)
        fwd_val           = inner_prod(adj_grad_out, d_in)
        ref_val           = inner_prod(ref_out[1], d_out)
        #print("OH HERE\n", fwd_val, ref_val)
        np.testing.assert_allclose(fwd_val, ref_val)

  def test_filterdown_adjoint(self):
    func                  = self.gen_func_memo()
    deriv_sig             = self.gen_deriv_sig()
    d_in_is_tup           = (len(deriv_sig) > 1)
    adj_func              = func._TEST_NIR_Adjoint(**deriv_sig)
    adj_func              = adj_func._TEST_NIR_filterdown()
    ref_dfunc             = self.gen_deriv_memo()
    print(adj_func)
    for fname,in_data,d_in,d_out in self.discover_rand_adjoint_inout():
      with self.subTest(data=fname):
        #print('D_IN\n',d_in)
        #print('D_OUT\n',d_out)
        adj_out           = adj_func.interpret(*( in_data+(d_out,) ))
        ref_out           = ref_dfunc.interpret(*(in_data+d_in))
        #print("*\n*\n*\n*\n")
        adj_grad_out      = ( adj_out.grad_out if d_in_is_tup
                                             else (adj_out.grad_out,) )
        #print('adj_out\n',adj_out.out)
        #print('ref_out\n',ref_out[0])
        fwd_val           = inner_prod(adj_grad_out, d_in)
        ref_val           = inner_prod(ref_out[1], d_out)
        #print("OH HERE\n", fwd_val, ref_val)
        np.testing.assert_allclose(fwd_val, ref_val)

  #def test_print_perf_report(self):
  #  orig                  = self.gen_func_memo()
  #  opt                   = orig._TEST_NIR_Roundtrip_YesSimp()
  #  #print(orig)
  #  #print(opt)
  #  fname,in_data,out_data = self.discover_rand_datum()
  #  with self.subTest(data=fname):
  #    print(">> orig")
  #    orig.perf_counts(*in_data)
  #    print(">> NIR roundtrip")
  #    opt.perf_counts(*in_data)

  #def test_print_perf_report(self):
  #  orig                  = self.gen_func_memo()
  #  opt                   = orig._TEST_NIR_Roundtrip_YesSimp()
  #  deriv_sig             = self.gen_deriv_sig()
  #  deriv                 = orig._TEST_NIR_Deriv(**deriv_sig)
  #  fname,in_data,d_in_data = self.discover_rand_deriv_datum()
  #  with self.subTest(data=fname):
  #    print(">> nir-opt")
  #    opt.perf_counts(*in_data)
  #    print(">> deriv")
  #    deriv.perf_counts(*(in_data+d_in_data))

  def test_print_perf_report(self):
    orig                  = self.gen_func_memo()
    opt                   = orig._TEST_NIR_Roundtrip_YesSimp()
    deriv_sig             = self.gen_deriv_sig()
    #deriv                 = orig._TEST_NIR_Deriv(**deriv_sig)
    adj_func              = orig._TEST_NIR_Adjoint(**deriv_sig)
    adj_func              = adj_func._TEST_NIR_filterdown()
    filterdown            = orig._TEST_NIR_filterdown()
    print(filterdown)
    #fname,in_data,d_in_data = self.discover_rand_deriv_datum()
    fname,in_data,d_in,d_out = self.discover_rand_adjoint_inout()[0]
    with self.subTest(data=fname):
      print(">> orig")
      orig.perf_counts(*in_data)
      print(">> nir-opt")
      opt.perf_counts(*in_data)
      print(">> nir-filterdown")
      filterdown.perf_counts(*in_data)
      print(">> adjoint")
      adj_func.perf_counts(*( in_data+(d_out,) ))
  

  def test_wallclock_perf(self):
    orig                  = self.gen_func_memo()
    opt                   = orig._TEST_NIR_Roundtrip_YesSimp()
    deriv_sig             = self.gen_deriv_sig()
    #deriv                 = orig._TEST_NIR_Deriv(**deriv_sig)
    adj_func              = orig._TEST_NIR_Adjoint(**deriv_sig)
    adj_func              = adj_func._TEST_NIR_filterdown()
    filterdown            = orig._TEST_NIR_filterdown()
    #print(opt)

    def do_timing(f, args, n_runs=10):
      # make sure it's jit-ed
      out_buf = f(*args)
      def build_out(out_buf):
        if type(out_buf) is tuple:
          return tuple( build_out(t) for t in out_buf )
        elif isinstance(out_buf, tuple):
          return type(out_buf)(*[ build_out(t) for t in out_buf ])
        elif type(out_buf) == float:
          return self.rand.rand_ndarray((1,))
        else:
          return out_buf
      out_buf = build_out(out_buf)

      lo, hi, avg = 1.0e6, 0.0, 0.0
      for i in range(0,n_runs):
        start   = time.perf_counter()
        f(*args, output=out_buf)
        stop    = time.perf_counter()

        dt = stop-start
        lo = lo if lo < dt else dt
        hi = hi if hi > dt else dt
        avg += dt

      print(f"  lo: {lo}  hi: {hi}  avg: {avg/n_runs}")
      return { 'lo':lo, 'hi':hi, 'avg':avg/n_runs }

    fname,in_data,d_in,d_out = self.discover_rand_perf_inout()[0]
    with self.subTest(data=fname):
      print(">> orig")
      do_timing(orig, in_data)
      print(">> nir-opt")
      do_timing(opt, in_data)
      print(">> nir-filterdown")
      do_timing(filterdown, in_data)
      print(">> adjoint")
      do_timing(adj_func, in_data+(d_out,) )

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# A helper for random numbers
class RandKey:
  def to_local_state(self):
    self._global_state    = random.getstate()
    random.setstate(self._local_state)
  def to_global_state(self):
    self._local_state     = random.getstate()
    random.setstate(self._global_state)

  def __init__(self, seed=0):
    # seed = 0 is chosen to ensure reproducibility in tests by default
    self._global_state    = random.getstate()
    random.seed(seed)
    self._seed            = seed
    self.to_global_state()

  def rand_ndarray( self, shape ):
    arr = np.asfortranarray(np.random.random_sample(size=shape))
    arr = 4.0*arr - 2.0
    #arr = np.zeros(shape, order='F')
    #with np.nditer(arr, op_flags=['readwrite']) as it:
    #  for x in it:
    #    x[...] = gen(self)
    return arr

  def rand_bool_array( self, shape ):
    arr = np.random.choice(a=[True,False], size=shape, p=[0.5,0.5])
    arr = np.asfortranarray(arr)
    #np.ndarray(shape, dtype=bool, order='F')
    #with np.nditer(arr, op_flags=['readwrite']) as it:
    #  for x in it:
    #    x[...] = self.choice([True,False])
    return arr

# patch methods from random into the RandKey class
def patch_method(fname,f):
  @wraps(f)
  def wrapper(self, *args, **kwargs):
    self.to_local_state()
    result = f(*args, **kwargs)
    self.to_global_state()
    return result
  setattr(RandKey,fname,wrapper)

for fname in [
  'randrange', 'randint', 'choice', 'shuffle',
  'sample', 'random', 'uniform'
]:
  f = getattr(random,fname)
  patch_method(fname,f)

# --------------------------------------------------------------------------- #

def inner_prod(lhs,rhs):
  if type(lhs) is np.ndarray:
    assert type(rhs) is np.ndarray
    return np.tensordot(lhs, rhs, len(lhs.shape))
  elif type(lhs) == float or type(lhs) == np.float64:
    assert type(rhs) == float or type(rhs) == np.float64
    return lhs*rhs
  elif isinstance(lhs, tuple):
    assert isinstance(rhs, tuple)
    assert len(lhs) == len(rhs)
    return sum([ inner_prod(L,R) for L,R in zip(lhs,rhs) ])
  else: assert False, f"unexpected type: {type(lhs)}"

