
import unittest
from ATL.halide_ir import HIR
import time

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class TestHalideWrapper(unittest.TestCase):

  def gen_checker_arr(self,w,h,k):
    a = []
    for x in range(0,w):
      a.append([])
      x_1       = (x//k)%2
      for y in range(0,h):
        y_1     = (y//k)%2
        a[x].append( float((x_1+y_1)%2) )
    return np.array(a, order='F')

  def gen_ramp_arr(self,w,h):
    a = []
    for x in range(0,w):
      a.append([])
      for y in range(0,h):
        a[x].append( float(x+y) )
    return np.array(a, order='F')


  def gen_blur(self):
    in_img      = HIR.Img('in_img', 2, HIR.f64)
    InImg       = HIR.ImgFunc(in_img)
    w, h        = HIR.Param('w', HIR.i32), HIR.Param('h', HIR.i32)
    W, H        = HIR.Eparam(w), HIR.Eparam(h)

    clamp       = lambda x,lo,hi: HIR.Max( lo, HIR.Min( hi, x ) )
    i0          = HIR.Const(0,HIR.i32)
    i1          = HIR.Const(1,HIR.i32)
    m1          = lambda x: HIR.BinOp('-',x,i1)
    p1          = lambda x: HIR.BinOp('+',x,i1)

    x, y        = HIR.Var('x'), HIR.Var('y')
    X, Y        = HIR.Evar(x), HIR.Evar(y)
    f           = HIR.Func('f')
    blur_x      = HIR.Func('blur_x')
    blur_y      = HIR.Func('blur_y')

    stmts       = [
      HIR.PureDef( f, [x,y], HIR.FAccess(InImg, [
                        clamp(X,i0,m1(W)), clamp(Y,i0,m1(H)) ])),
      HIR.PureDef( blur_x, [x,y],
                   HIR.BinOp('/',
                    HIR.BinOp('+',
                      HIR.BinOp('+',
                        HIR.FAccess(f,[m1(X),Y]),
                        HIR.BinOp('*',HIR.Const(2.0,HIR.f64),
                          HIR.FAccess(f,[X,Y]))),
                        HIR.FAccess(f,[p1(X),Y])),
                    HIR.Const(4.0,HIR.f64))),
      HIR.PureDef( blur_y, [x,y],
                   HIR.BinOp('/',
                    HIR.BinOp('+',
                      HIR.BinOp('+',
                        HIR.FAccess(blur_x,[X,m1(Y)]),
                        HIR.BinOp('*',HIR.Const(2.0,HIR.f64),
                          HIR.FAccess(blur_x,[X,Y]))),
                        HIR.FAccess(blur_x,[X,p1(Y)])),
                    HIR.Const(4.0,HIR.f64))),
    ]

    return HIR.Pipeline(
              vars      = [x,y],
              rdoms     = [],
              funcs     = [f,blur_x,blur_y],
              params    = [w,h],
              imgs      = [in_img],
              stmts     = stmts,
              outputs   = [blur_y] )

  def test_blur(self):
    checker     = self.gen_checker_arr(100,100,10)
    ramp        = self.gen_ramp_arr(100,100)

    blur_out    = np.zeros([100,100], order='F')

    blurpipe    = self.gen_blur()

    with self.subTest(data="ramp"):
      blurpipe([100,100],[ramp],[blur_out])
      predict = ramp.copy()
      for y in range(0,100):
        predict[0,y]  += 0.25
        predict[99,y] -= 0.25
      for x in range(0,100):
        predict[x,0]  += 0.25
        predict[x,99] -= 0.25

      np.testing.assert_allclose(predict, blur_out)

    with self.subTest(data="checker"):
      blurpipe([100,100],[checker],[blur_out])
      predict = checker.copy()
      K       = np.zeros([3,3])
      for x in range(0,100):
        for y in range(0,100):
          xlo, xhi  = (0 if x>0 else 1), (2 if x<99 else 1)
          ylo, yhi  = (0 if y>0 else 1), (2 if y<99 else 1)
          K[xlo:xhi+1,ylo:yhi+1] = checker[x+xlo-1:x+xhi,y+ylo-1:y+yhi]
          if x == 0:
            K[0,0:3] = K[1,0:3]
          elif x == 99:
            K[2,0:3] = K[1,0:3]
          if y == 0:
            K[0:3,0] = K[0:3,1]
          elif y == 99:
            K[0:3,2] = K[0:3,1]

          K[1,0:3] = 0.25*K[0,0:3] + 0.5*K[1,0:3] + 0.25*K[2,0:3]
          predict[x,y] = 0.25*K[1,0] + 0.5*K[1,1] + 0.25*K[1,2]

      np.testing.assert_allclose(predict, blur_out)
    

  def gen_laplacian(self):
    f           = HIR.Img('f', 2, HIR.f64)
    F           = HIR.ImgFunc(f)
    w, h        = HIR.Param('w', HIR.i32), HIR.Param('h', HIR.i32)
    W, H        = HIR.Eparam(w), HIR.Eparam(h)

    clamp       = lambda x,lo,hi: HIR.Max( lo, HIR.Min( hi, x ) )
    i0          = HIR.Const(0,HIR.i32)
    i1          = HIR.Const(1,HIR.i32)
    f0          = HIR.Const(0.0,HIR.f64)
    m1          = lambda x: HIR.BinOp('-',x,i1)
    p1          = lambda x: HIR.BinOp('+',x,i1)
    Eq          = lambda x,y: HIR.BinOp('==',x,y)

    x           = HIR.RDom('x', [HIR.Range(i0,W)])
    y           = HIR.RDom('y', [HIR.Range(i0,H)])
    X, Y        = HIR.Erdom(x), HIR.Erdom(y)
    i, j        = HIR.Var('i'), HIR.Var('j')
    I, J        = HIR.Evar(i), HIR.Evar(j)

    dx, dy      = HIR.Func('dx'), HIR.Func('dy')
    result      = HIR.Func('result')

    sumbody     = HIR.BinOp('+',
                    HIR.Select( Eq(X,m1(W)), f0,
                      HIR.BinOp('*', HIR.FAccess(dx,[X,Y]),
                                     HIR.FAccess(dx,[X,Y]) )),
                    HIR.Select( Eq(Y,m1(H)), f0,
                      HIR.BinOp('*', HIR.FAccess(dy,[X,Y]),
                                     HIR.FAccess(dy,[X,Y]) ))
                  )

    stmts       = [
      HIR.PureDef( dx, [i,j], HIR.BinOp('-',
                    HIR.FAccess(F,[HIR.Min(p1(I),m1(W)),J]),
                    HIR.FAccess(F,[I,J])) ),
      HIR.PureDef( dy, [i,j], HIR.BinOp('-',
                    HIR.FAccess(F,[I,HIR.Min(p1(J),m1(H))]),
                    HIR.FAccess(F,[I,J])) ),
      HIR.PureDef( result, [i], f0 ),
      HIR.Update( result, [i0], HIR.BigSum(x, HIR.BigSum(y, sumbody)) ),
    ]

    return HIR.Pipeline(
              vars      = [i,j],
              rdoms     = [x,y],
              funcs     = [result,dx,dy],
              params    = [w,h],
              imgs      = [f],
              stmts     = stmts,
              outputs   = [result] )

  def test_laplacian(self):
    checker     = self.gen_checker_arr(100,100,10)
    ramp        = self.gen_ramp_arr(100,100)

    energy_out  = np.zeros([1], order='F')

    laplacepipe = self.gen_laplacian()

    with self.subTest(data="ramp"):
      laplacepipe([100,100],[ramp],[energy_out])
      predict   = 2*100.0*100.0 - 2*100.0
      np.testing.assert_allclose(predict, energy_out[0])

    with self.subTest(data="checker"):
      laplacepipe([100,100],[checker],[energy_out])
      predict   = 2*9*100.0
      np.testing.assert_allclose(predict, energy_out[0])


  def gen_grad(self):
    f           = HIR.Img('f', 2, HIR.f64)
    F           = HIR.ImgFunc(f)

    i0          = HIR.Const(0,HIR.i32)
    i1          = HIR.Const(1,HIR.i32)
    f0          = HIR.Const(0.0,HIR.f64)
    m1          = lambda x: HIR.BinOp('-',x,i1)
    p1          = lambda x: HIR.BinOp('+',x,i1)

    i, j        = HIR.Var('i'), HIR.Var('j')
    I, J        = HIR.Evar(i), HIR.Evar(j)

    dx, dy      = HIR.Func('dx'), HIR.Func('dy')

    stmts       = [
      HIR.PureDef( dx, [i,j], HIR.BinOp('-',
                    HIR.FAccess(F,[p1(I),J]),
                    HIR.FAccess(F,[I,J])) ),
      HIR.PureDef( dy, [i,j], HIR.BinOp('-',
                    HIR.FAccess(F,[I,p1(J)]),
                    HIR.FAccess(F,[I,J])) ),
    ]

    return HIR.Pipeline(
              vars      = [i,j],
              rdoms     = [],
              funcs     = [dx,dy],
              params    = [],
              imgs      = [f],
              stmts     = stmts,
              outputs   = [dx,dy] )

  def test_grad(self):
    checker     = self.gen_checker_arr(100,100,10)
    ramp        = self.gen_ramp_arr(100,100)

    grad_x_out  = np.zeros([99,100], order='F')
    grad_y_out  = np.zeros([100,99], order='F')
    predict_x   = grad_x_out.copy()
    predict_y   = grad_y_out.copy()

    gradpipe    = self.gen_grad()

    with self.subTest(data="ramp"):
      gradpipe([],[ramp],[grad_x_out,grad_y_out])

      predict_x[0:99,0:100] = 1.0
      predict_y[0:100,0:99] = 1.0

      np.testing.assert_allclose(predict_x, grad_x_out)
      np.testing.assert_allclose(predict_y, grad_y_out)

    with self.subTest(data="checker"):
      gradpipe([],[checker],[grad_x_out,grad_y_out])

      predict_x[0:99,0:100] = 0.0
      predict_y[0:100,0:99] = 0.0
      for x in range(9,99,10):
        for y in range(0,100):
          predict_x[x,y]    = 1.0 if checker[x,y] == 0.0 else -1.0
      for x in range(0,100):
        for y in range(9,99,10):
          predict_y[x,y]    = 1.0 if checker[x,y] == 0.0 else -1.0

      np.testing.assert_allclose(predict_x, grad_x_out)
      np.testing.assert_allclose(predict_y, grad_y_out)




