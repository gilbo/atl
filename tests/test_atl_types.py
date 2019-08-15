
import unittest
from ATL import atl_types as T

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Tests

class Test_ATL_Types(unittest.TestCase):

  def test_istype(self):
    nn    = T.Tuple(None,[T.num,T.num])
    ab    = T.Tuple(T.labels(['a','b']),[T.num,T.num])
    vec3  = T.Tensor(3,T.num)
    self.assertTrue( T.is_type(T.num) )
    self.assertTrue( T.is_type(T.error) )
    self.assertTrue( T.is_type(nn) )
    self.assertTrue( T.is_type(nn) )
    self.assertTrue( T.is_type(vec3) )

  def test_memo(self):
    nn    = T.Tuple(None,[T.num,T.num])
    ab    = T.Tuple(T.labels(['a','b']),[T.num,T.num])
    vec3  = T.Tensor(3,T.num)
    self.assertEqual(T.num,    T.Num())
    self.assertEqual(T.error,  T.Error())
    self.assertEqual(nn,    T.Tuple(None,[T.num,T.num]))
    self.assertEqual(vec3,  T.Tensor(3,T.num))
    self.assertEqual(ab,    T.Tuple(T.labels(['a','b']),[T.num,T.num]))
    self.assertNotEqual(ab, nn)

  def test_SoA(self):
    nn    = T.Tuple(None,[T.num,T.num])
    vec3  = T.Tensor(3,T.num)
    nn3   = T.Tensor(3,nn)
    v3v3  = T.Tuple(None,[vec3,vec3])
    self.assertTrue(   nn.has_tuples()     )
    self.assertFalse(  vec3.has_tuples()   )
    self.assertTrue(   nn3.has_tuples()    )
    self.assertTrue(   v3v3.has_tuples()   )

    self.assertTrue(   nn.is_SoA()    )
    self.assertTrue(   vec3.is_SoA()  )
    self.assertFalse(  nn3.is_SoA()   )
    self.assertTrue(   v3v3.is_SoA()  )
    self.assertTrue(   nn3.SoA_transform().is_SoA()   )

  def test_matches(self):
    nn      = T.Tuple(None,[T.num,T.num])
    vec3    = T.Tensor(3,T.num)
    nn3     = T.Tensor(3,nn)
    v3v3    = T.Tuple(None,[vec3,vec3])
    test    = T.Tuple(None,[T.num,vec3])
    abtest  = T.Tuple(T.labels(['a','b']),[T.num,vec3])
    cbtest  = T.Tuple(T.labels(['c','b']),[T.num,vec3])

    self.assertTrue(    v3v3.matches(nn3.SoA_transform())   )
    self.assertTrue(    test.matches(abtest)      )
    self.assertFalse(   cbtest.matches(abtest)    )
    self.assertTrue(    cbtest.matches(test)      )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
  unittest.main()
