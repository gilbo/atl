
import unittest
from ATL import *

import numpy as np
import random

from functools import wraps
from collections import namedtuple

from re import compile as _re_compile
_data_method_RE = _re_compile("^data_")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

DataCase  = namedtuple('DataCase',['name','input','output'])
InputCase = namedtuple('InputCase',['name','input'])

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

  def gen_deriv(self):
    raise NotImplementedError()

  def gen_deriv_sig(self):
    raise NotImplementedError()

  def rand_input(self):
    raise NotImplementedError()

  def rand_deriv_input(self):
    raise NotImplementedError()

  def n_rand_tests(self):
    return 2

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
      data_cases  = self.discover_data().copy()
      n_default   = self.n_rand_tests() - len(data_cases)
      func        = self.gen_func()

      for i in range(0,n_default):
        in_data   = self.rand_input()
        out_data  = func.interpret(*in_data)
        data_cases.append( DataCase(f"rand_{i}",in_data,out_data) )

      self._ftc_rand_data = data_cases

    return self._ftc_rand_data

  def discover_rand_deriv_input(self):
    if not hasattr(self, '_ftc_rand_input_data'):
      self._init_rand()

      data_cases  = []
      n_default   = self.n_rand_tests()

      for i in range(0,n_default):
        in_data   = self.rand_deriv_input()
        data_cases.append( InputCase(f"rand_deriv_{i}",in_data) )

      self._ftc_rand_input_data = data_cases

    return self._ftc_rand_input_data

  def test_print(self):
    self.gen_func()

  def test_print(self):
    deriv_sig         = self.gen_deriv_sig()
    dfunc             = self.gen_func()._TEST_TotalDeriv_Alone(**deriv_sig)
    #print(dfunc)

  def test_interpreter(self):
    for fname,in_data,out_data in self.discover_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        comp_out          = func.interpret(*in_data)
        np.testing.assert_allclose(comp_out, out_data)

  def test_let_lift(self):
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        liftfunc          = func._TEST_LetLift()
        lift_out          = func.interpret(*in_data)
        np.testing.assert_allclose(out_data, lift_out)

  def test_tuple_elim(self):
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        liftfunc          = func._TEST_TupleElimination()
        lift_out          = func.interpret(*in_data)
        np.testing.assert_allclose(out_data, lift_out)

  def test_pre_normalization(self):
    for fname,in_data,out_data in self.discover_rand_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        liftfunc          = func._TEST_PreNormalization()
        lift_out          = func.interpret(*in_data)
        np.testing.assert_allclose(out_data, lift_out)

  def test_total_derivative_alone(self):
    for fname,in_data in self.discover_rand_deriv_input():
      with self.subTest(data=fname):
        func              = self.gen_func()
        deriv_sig         = self.gen_deriv_sig()
        dfunc             = func._TEST_TotalDeriv_Alone(**deriv_sig)
        ref_dfunc         = self.gen_deriv()

        df_out            = dfunc(*in_data)
        ref_out           = ref_dfunc(*in_data)
        np.testing.assert_allclose(df_out, ref_out)

  def test_derivative(self):
    for fname,in_data in self.discover_rand_deriv_input():
      with self.subTest(data=fname):
        func              = self.gen_func()
        deriv_sig         = self.gen_deriv_sig()
        dfunc             = func._TEST_TotalDeriv(**deriv_sig)
        ref_dfunc         = self.gen_deriv()

        df_out            = dfunc(*in_data)
        ref_out           = ref_dfunc(*in_data)
        np.testing.assert_allclose(df_out, ref_out)

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

  def rand_ndarray( self, shape, gen=(lambda self: self.uniform(-2,2)) ):
    arr = np.zeros(shape)
    with np.nditer(arr, op_flags=['readwrite']) as it:
      for x in it:
        x[...] = gen(self)
    return arr

  def rand_bool_array( self, shape ):
    arr = np.ndarray(shape, dtype=bool)
    with np.nditer(arr, op_flags=['readwrite']) as it:
      for x in it:
        x[...] = self.choice([True,False])
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


