
import unittest
from ATL import *

import numpy as np

from re import compile as _re_compile
_data_method_RE = _re_compile("^data_")

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class FunctionTestCase:
  """ A testing mixin that should be used alongside unittest.TestCase
  Expects gen_func() to be defined returning an ATL function.  Then every
  method with name data_*() is expected to return a sample input/output
  data pair for testing the function with.
  """

  def gen_func(self):
    raise NotImplementedError()

  def discover_data(self):
    data_fstrs = [ f for f in dir(self)
                     if _data_method_RE.match(f) != None ]
    data_funcs = [ (f, getattr(self, f)) for f in data_fstrs ]
    return data_funcs

  def test_print(self):
    pass #print( self.gen_func() )

  #def test_print_letlift(self):
  #  print( self.gen_func()._TEST_LetLift() )

  def test_interpreter(self):
    for fname,gen_data in self.discover_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        in_data, out_data = gen_data()
        comp_out          = func.interpret(*in_data)
        np.testing.assert_allclose(comp_out, out_data)

  def test_let_lift(self):
    for fname,gen_data in self.discover_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        in_data, out_data = gen_data()
        comp_out          = func.interpret(*in_data)

        liftfunc          = func._TEST_LetLift()
        lift_out          = func.interpret(*in_data)

        np.testing.assert_allclose(comp_out, lift_out)

  def test_tuple_elim(self):
    for fname,gen_data in self.discover_data():
      with self.subTest(data=fname):
        func              = self.gen_func()
        in_data, out_data = gen_data()
        comp_out          = func.interpret(*in_data)

        liftfunc          = func._TEST_TupleElimination()
        lift_out          = func.interpret(*in_data)

        np.testing.assert_allclose(comp_out, lift_out)








