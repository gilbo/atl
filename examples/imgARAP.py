from __future__ import annotations
import sys
sys.path += '..'
import ATL
from ATL import num

import numpy as np
import time
import random
from PIL import Image



def gen_func():
  @ATL.func
  def Rot2D( ang : num, v2 : num[2] ):
    s   = sin(ang)
    c   = cos(ang)
    return [ c*v2[0] - s*v2[1],
             s*v2[0] + c*v2[1] ]

  @ATL.macro
  def regular(i,j,di,dj,P):
    d_off[k:2]  = P.Offsets[i,j,k] - P.Offsets[i+di,j+dj,k]
    d_ur[k:2]   = P.UrShape[i,j,k] - P.UrShape[i+di,j+dj,k]
    rot         = Rot2D( P.Angle[i,j], d_ur )
    diff[k:2]   = d_off[k] - rot[k]
    return Sum[k:2]( diff[k]*diff[k] )

  @ATL.func
  def E_ARAP(
    # image size
    W : size, H : size,
    # energy term weights
    w_fit : num, w_reg : num,
    # unknown vector fields
    Offsets : num[W,H,2], Angle : num[W,H],
    # input (known) vector fields
    UrShape : num[W,H,2], Constraints : num[W,H,2],
    # masking relations
    C_valid : (W,H), Mask : (W,H)
  ):
    P               = { Offsets:Offsets, UrShape:UrShape, Angle:Angle }
    E_reg[i:W,j:H]  = ( (i+1<W) * (Mask(i+1,j) * regular(i,j, 1, 0, P)) +
                       (i-1>=0) * (Mask(i-1,j) * regular(i,j,-1, 0, P)) +
                        (j+1<H) * (Mask(i,j+1) * regular(i,j, 0, 1, P)) +
                       (j-1>=0) * (Mask(i,j-1) * regular(i,j, 0,-1, P)) )
    OC[i:W,j:H,k:2] = Offsets[i,j,k] - Constraints[i,j,k]
    E_fit[i:W,j:H]  = Sum[k:2]( OC[i,j,k]*OC[i,j,k] )
    return Sum[i:W,j:H](Mask(i,j)*(
                          w_fit * (C_valid(i,j) * E_fit[i,j])
                        + w_reg * E_reg[i,j] ))
  
  return E_ARAP

E_ARAP  = gen_func()

# --------------------------------------------------------------------------- #

print(E_ARAP)

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



def time_for_img(filename,niters=10):
  mask        = read_img_mask(filename)
  W,H         = mask.shape # swapped b/c Fortran ordering
  w_fit       = 1.0
  w_reg       = 1.0e-3
  Offsets     = np.asfortranarray(np.random.rand(W,H,2))
  Angle       = np.asfortranarray(np.random.rand(W,H))
  UrShape     = np.asfortranarray(np.random.rand(W,H,2))
  Constraints = np.asfortranarray(np.random.rand(W,H,2))
  C_valid     = gen_C_valid(mask)
  scalar      = np.zeros([1],order='F')

  dOffsets    = np.asfortranarray(np.random.rand(W,H,2))
  dAngle      = np.asfortranarray(np.random.rand(W,H))
  # prime the pump and cause compilation etc.
  E_ARAP(W,H, w_fit,w_reg, Offsets,Angle,
         UrShape, Constraints, C_valid, mask, output=scalar)



# get script directory
import os
ex_dir = os.path.dirname(os.path.realpath(__file__))


time_for_img(ex_dir+"/mask0.png")








