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

import time
import random
from PIL import Image

import jax
import jax.numpy as np
#import skimage.io

from global_params import *




def gen_funcs():

  def Rot2D( ang, v2 ):
    s       = np.sin(ang)
    c       = np.cos(ang)
    return np.stack([ c * v2[:,:,0] - s * v2[:,:,1],
                      s * v2[:,:,0] - c * v2[:,:,1]], axis=-1)

  def regular( off, off_shft, ur, ur_shft, ang ):
    # hello
    d_off   = off - off_shft
    d_ur    = ur  - ur_shft
    rot     = Rot2D( ang, d_ur )
    diff    = 0.5 * (d_off - rot)
    return diff

  def res_ARAP(
    # image size
    W, H,
    # unknown vector fields
    Offsets, Angle,
    # input (known) vector fields
    UrShape, Constraints,
    # masking relations
    C_valid, Mask
  ):
    Offsets_left = np.roll(Offsets, shift=-1, axis=0)
    Offsets_right = np.roll(Offsets, shift=1, axis=0)
    Offsets_up = np.roll(Offsets, shift=-1, axis=1)
    Offsets_down = np.roll(Offsets, shift=1, axis=1)

    UrShape_left = np.roll(UrShape, shift=-1, axis=0)
    UrShape_right = np.roll(UrShape, shift=1, axis=0)
    UrShape_up = np.roll(UrShape, shift=-1, axis=1)
    UrShape_down = np.roll(UrShape, shift=1, axis=1)

    Mask_left = np.roll(Mask, shift=-1, axis=0)
    Mask_right = np.roll(Mask, shift=1, axis=0)
    Mask_up = np.roll(Mask, shift=-1, axis=1)
    Mask_down = np.roll(Mask, shift=1, axis=1)

    ML = np.repeat(np.reshape(np.logical_and(Mask, Mask_left),[W,H,1]),
                   repeats=2,axis=2)
    MR = np.repeat(np.reshape(np.logical_and(Mask, Mask_right),[W,H,1]),
                   repeats=2,axis=2)
    MU = np.repeat(np.reshape(np.logical_and(Mask, Mask_up),[W,H,1]),
                   repeats=2,axis=2)
    MD = np.repeat(np.reshape(np.logical_and(Mask, Mask_down),[W,H,1]),
                   repeats=2,axis=2)

    #print(np.logical_and(Mask, Mask_left).dtype)
    
    Ereg_left   = ( ML * regular(Offsets, Offsets_left,
                                 UrShape, UrShape_left, Angle) )
    Ereg_right  = ( MR * regular(Offsets, Offsets_right,
                                 UrShape, UrShape_right, Angle) )
    Ereg_up     = ( MU * regular(Offsets, Offsets_up,
                                 UrShape, UrShape_up, Angle) )
    Ereg_down   = ( MD * regular(Offsets, Offsets_down,
                                 UrShape, UrShape_up, Angle) )

    MC = np.repeat(np.reshape(np.logical_and(Mask, C_valid),[W,H,1]),
                   repeats=2,axis=2)

    Efit        = (MC * 0.5 * (Offsets - Constraints))

    return ( Efit,
             Ereg_left,
             Ereg_right,
             Ereg_up,
             Ereg_down ) # add axis?

  def JtJ_ARAP(
    # image size
    W, H,
    # unknown vector fields
    Offsets, Angle,
    dOffsets, dAngle,
    # input (known) vector fields
    UrShape, Constraints,
    # masking relations
    C_valid, Mask
  ):
    def F(off, ang):
      return res_ARAP(W, H, off, ang, UrShape, Constraints, C_valid, Mask)

    _,  Jx  = jax.jvp(F, [Offsets, Angle], [dOffsets, dAngle])
    _, Jt   = jax.vjp(F, Offsets, Angle)
    return Jt(Jx)

  #res_ARAP = jax.jit(res_ARAP)
  #JtJ_ARAP = jax.jit(JtJ_ARAP)
  
  return (res_ARAP, JtJ_ARAP)



# --------------------------------------------------------------------------- #

#res_ARAP, JtJ_ARAP  = gen_funcs()
#print(res_ARAP, JtJ_ARAP)

# --------------------------------------------------------------------------- #



def read_img_mask(filename):
  img   = Image.open(filename)
  I     = np.transpose(np.array(img))#(np.asfortranarray(img))
  mask  = np.array( np.less( np.zeros(I.shape), I ), dtype = np.uint8 )
  return mask

def gen_C_valid(mask):
  shape       = mask.shape
  C_valid     = np.zeros(shape,dtype=np.uint8)

  for k in range(0,5):
    success   = False
    while not success:
      i       = random.randint(0,shape[0]-1)
      j       = random.randint(0,shape[1]-1)
      if mask[i,j]:
        C_valid = jax.ops.index_update(C_valid,(i,j),True)
        break

  return C_valid


def get_time(mask_no,n_iters=default_timing_iters,timeout_sec=10):
  JtJtimes  = None
  Rtimes    = None

  filename = str(os.path.join(_PERF_DIR,f'mask{mask_no}.png'))

  # load the mask
  mask      = read_img_mask(filename)
  C_valid   = gen_C_valid(mask)

  # inputs
  key = jax.random.PRNGKey(1234)

  W,H         = mask.shape # swapped b/c Fortran ordering
  Offsets     = jax.random.uniform(key, shape=[W,H,2],  dtype=np.float64)
  Angle       = jax.random.uniform(key, shape=[W,H],    dtype=np.float64)
  UrShape     = jax.random.uniform(key, shape=[W,H,2],  dtype=np.float64)
  Constraints = jax.random.uniform(key, shape=[W,H,2],  dtype=np.float64)

  dOffsets    = jax.random.uniform(key, shape=[W,H,2],  dtype=np.float64)
  dAngle      = jax.random.uniform(key, shape=[W,H],    dtype=np.float64)

  out_Off     = jax.random.uniform(key, shape=[W,H,2],  dtype=np.float64)
  out_Ang     = jax.random.uniform(key, shape=[W,H],    dtype=np.float64)

  #residuals = tuple( jax.random.uniform(key, shape=[W,H,2], dtype=np.float64)
  #                   for _ in range(0,5) )

  # functions
  res_ARAP, JtJ_ARAP = gen_funcs()

  # make sure at least O(100 M) operations are executed
  # so that the timing base-line is at least the order of 0.1 sec
  #if W * H * n_iters < 1e7:
  #  n_iters = int(1e7)//(N*N)

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def Rtimes():
    residuals = res_ARAP( W,H,
                          Offsets, Angle,
                          UrShape, Constraints,
                          C_valid, mask )

  @take_time_w_timeout(n_iters=n_iters, timeout_sec=timeout_sec)
  def JtJtimes():
    dout = JtJ_ARAP( W,H,
                     Offsets, Angle,
                     dOffsets, dAngle,
                     UrShape, Constraints,
                     C_valid, mask )

  return (W,H,Rtimes,JtJtimes)


if __name__ == "__main__":
  print("imgARAP timings for ATL")

  for mask_no in [0,1,2]:
    W, H, Rtimes, JtJtimes = get_time(mask_no)
    #basetime, dtime   = get_time(N,n_iters=n_iters)
    Rtimes    = 1e3 * (Rtimes.avg() or math.inf)
    JtJtimes  = 1e3 * (JtJtimes.avg() or math.inf)
    print( f"{W:4d}, {H:4d}: {Rtimes:8.3f} ms    {JtJtimes:8.3f} ms "
           f"  {JtJtimes/Rtimes:8.3f}" )





