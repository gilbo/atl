from __future__ import annotations

import os
import sys
_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR       = os.path.abspath(os.path.join(_HERE_DIR,"..",".."))
_PERF_DIR       = os.path.abspath(os.path.join(_HERE_DIR,".."))
if _BASE_DIR not in sys.path:
  sys.path.append(_BASE_DIR)
if _PERF_DIR not in sys.path:
  sys.path.append(_PERF_DIR)

import ATL
from ATL import num

import numpy as np
import math
import random
from PIL import Image

from global_params import *



def gen_funcs():
  @ATL.func
  def Rot2D( ang : num, v2 : num[2] ):
    s   = sin(ang)
    c   = cos(ang)
    return [ c*v2[0] - s*v2[1],
             s*v2[0] + c*v2[1] ]

  @ATL.macro
  def regular(i,j,di,dj,P, w_reg):
    d_off[k:2]  = P.Offsets[i,j,k] - P.Offsets[i+di,j+dj,k]
    d_ur[k:2]   = P.UrShape[i,j,k] - P.UrShape[i+di,j+dj,k]
    rot         = Rot2D( P.Angle[i,j], d_ur )
    diff[k:2]   = w_reg * (d_off[k] - rot[k])
    return diff

  @ATL.func
  def residuals_ARAP(
    # image size
    W : size, H : size,
    # unknown vector fields
    Offsets : num[W,H,2], Angle : num[W,H],
    # input (known) vector fields
    UrShape : num[W,H,2], Constraints : num[W,H,2],
    # masking relations
    C_valid : (W,H), Mask : (W,H)
  ):
    P                 = { Offsets:Offsets, UrShape:UrShape, Angle:Angle }
    E_reg_p0[i:W,j:H,k:2] = (i+1<W) * (Mask(i+1,j) * (Mask(i,j) *
                                   regular(i,j, 1, 0, P, 0.5)[k]))
    E_reg_n0[i:W,j:H,k:2] =(i-1>=0) * (Mask(i-1,j) * (Mask(i,j) *
                                   regular(i,j,-1, 0, P, 0.5)[k]))
    E_reg_0p[i:W,j:H,k:2] = (j+1<H) * (Mask(i,j+1) * (Mask(i,j) *
                                   regular(i,j, 0, 1, P, 0.5)[k]))
    E_reg_0n[i:W,j:H,k:2] =(j-1>=0) * (Mask(i,j-1) * (Mask(i,j) *
                                   regular(i,j, 0,-1, P, 0.5)[k]))
    OC[i:W,j:H,k:2]   = Mask(i,j) * (C_valid(i,j) *
                        (0.5 * (Offsets[i,j,k] - Constraints[i,j,k])))
    #E_fit[i:W,j:H]  = Sum[k:2]( OC[i,j,k]*OC[i,j,k] )
    return (E_reg_p0, E_reg_n0, E_reg_0p, E_reg_0n, OC)

  # define the JtJ form of the energy
  J   = residuals_ARAP.deriv(Offsets=True,Angle=True).proj(1)
  Jt  = residuals_ARAP.adjoint(Offsets=True,Angle=True,
                               output=('out','d_res')).proj(1)
  @ATL.func
  def JtJ_ARAP(
    # image size
    W : size, H : size,
    # unknown vector fields
    Offsets : num[W,H,2], Angle : num[W,H],
    dOffsets : num[W,H,2], dAngle : num[W,H],
    # input (known) vector fields
    UrShape : num[W,H,2], Constraints : num[W,H,2],
    # masking relations
    C_valid : (W,H), Mask : (W,H)
  ):
    Res   = J(W,H,
              Offsets, Angle,
              UrShape, Constraints,
              C_valid, Mask,
              dOffsets, dAngle)
    dOut  = Jt(W,H,
               Offsets, Angle,
               UrShape, Constraints,
               C_valid, Mask,
               Res)
    return dOut

  residuals_ARAP = residuals_ARAP.simplify()
  JtJ_ARAP = JtJ_ARAP.simplify()
  
  return (residuals_ARAP, JtJ_ARAP)


# --------------------------------------------------------------------------- #

#res_ARAP, JtJ_ARAP  = gen_funcs()
#print(res_ARAP, JtJ_ARAP)

# --------------------------------------------------------------------------- #

def read_img_mask(filename):
  img   = Image.open(filename)
  I     = np.transpose(np.asfortranarray(img))
  mask  = np.ndarray(I.shape,dtype=bool,order='F')
  for i in range(I.shape[0]):
    for j in range(I.shape[1]):
      mask[i,j] = (I[i,j] > 0)
  return mask

def gen_C_valid(mask):
  shape       = mask.shape
  C_valid     = np.ndarray(shape,dtype=bool,order='F')
  for i in range(0,shape[0]):
    for j in range(0,shape[1]):
      C_valid[i,j] = False

  for k in range(0,5):
    success   = False
    while not success:
      i       = random.randint(0,shape[0]-1)
      j       = random.randint(0,shape[1]-1)
      if mask[i,j]:
        C_valid[i,j] = True
        break

  return C_valid


def get_time(mask_no,n_iters=default_timing_iters,timeout_sec=10):
  JtJtimes  = None
  Rtimes    = None

  filename = str(os.path.join(_PERF_DIR,f'mask{mask_no}.png'))

  # load the mask
  mask      = read_img_mask(filename)
  C_valid   = gen_C_valid(mask)

  W,H         = mask.shape # swapped b/c Fortran ordering
  Offsets     = np.asfortranarray(np.random.rand(W,H,2))
  Angle       = np.asfortranarray(np.random.rand(W,H))
  UrShape     = np.asfortranarray(np.random.rand(W,H,2))
  Constraints = np.asfortranarray(np.random.rand(W,H,2))

  dOffsets    = np.asfortranarray(np.random.rand(W,H,2))
  dAngle      = np.asfortranarray(np.random.rand(W,H))

  out_Off     = np.asfortranarray(np.random.rand(W,H,2))
  out_Ang     = np.asfortranarray(np.random.rand(W,H))

  residuals   = tuple( np.asfortranarray(np.random.rand(W,H,2))
                       for _ in range(0,5) )

  # functions
  res_ARAP, JtJ_ARAP = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  #if W * H * n_iters < 1e7:
  #  n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Rtimes():
    res_ARAP( W,H,
              Offsets, Angle,
              UrShape, Constraints,
              C_valid, mask,
              output=residuals)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def JtJtimes():
    JtJ_ARAP( W,H,
              Offsets, Angle,
              dOffsets, dAngle,
              UrShape, Constraints,
              C_valid, mask,
              output=(out_Off, out_Ang))

  return (W,H,Rtimes,JtJtimes)


if __name__ == "__main__":
  print("imgARAP timings for ATL")
  ATL.set_backend('Halide')

  for mask_no in [0,1,2]:
    W, H, Rtimes, JtJtimes = get_time(mask_no)
    #basetime, dtime   = get_time(N,n_iters=n_iters)
    Rtimes    = 1e3 * (Rtimes.avg() or math.inf)
    JtJtimes  = 1e3 * (JtJtimes.avg() or math.inf)
    print( f"{W:4d}, {H:4d}: {Rtimes:8.3f} ms    {JtJtimes:8.3f} ms "
           f"  {JtJtimes/Rtimes:8.3f}" )




