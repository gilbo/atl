import os
import sys

_HERE_DIR       = os.path.abspath('')
sys.path += [str(_HERE_DIR)]

import pandas as pd
import math

# raw list of results...
RESULTS = []


def qform1():
  N_values = [100,200,400]#,800,1600,3200,6400]

  def do_run(compiler_str, get_time):
    for N in N_values:
      n_iters = 10 # make sure at least 10 iterations run
      basetime, htime   = get_time(N,n_iters=n_iters)
      basetime  = 1e3 * (basetime.avg() or math.inf)
      htime     = 1e3 * (htime.avg() or math.inf)
      print( f"{N:8d}: {basetime:8.3f} ms    {htime:8.3f} ms "
             f"  {htime/basetime:8.3f}" )
      RESULTS.append({ "problem":"qform1",
                       "prob_rettype":"scalar",
                       "compiler":compiler_str,
                       "N":N*N,
                       "f_ms_avg":basetime,
                       "Hf_ms_avg":htime,
                     })

  def ATL():
    from perf_ATL.qform1 import get_time
    do_run("ATL", get_time)
  ATL()

  def JAX():
    from perf_jax.qform1 import get_time
    do_run("JAX", get_time)
  JAX()

qform1()


def qform2():
  N_values = [100,200,400]#,800,1600,3200,6400]

  def do_run(compiler_str, get_time):
    for N in N_values:
      n_iters = 10 # make sure at least 10 iterations run
      basetime, dtime   = get_time(N,n_iters=n_iters)
      basetime  = 1e3 * (basetime.avg() or math.inf)
      dtime     = 1e3 * (dtime.avg() or math.inf)
      print( f"{N:8d}: {basetime:8.3f} ms    {dtime:8.3f} ms "
             f"  {dtime/basetime:8.3f}" )
      RESULTS.append({ "problem":"qform2",
                       "prob_rettype":"scalar",
                       "compiler":compiler_str,
                       "N":N*N,
                       "f_ms_avg":basetime,
                       "Df_ms_avg":dtime,
                     })

  def ATL():
    from perf_ATL.qform2 import get_time
    do_run("ATL",get_time)
  ATL()

  def JAX():
    from perf_jax.qform2 import get_time
    do_run("JAX", get_time)
  JAX()

qform2()



def img_ARAP():
  mask_vals = [0,1,2]

  def do_run(compiler_str, get_time):
    for mask_no in [0,1,2]:
      W, H, Rtimes, JtJtimes = get_time(mask_no)
      #basetime, dtime   = get_time(N,n_iters=n_iters)
      Rtimes    = 1e3 * (Rtimes.avg() or math.inf)
      JtJtimes  = 1e3 * (JtJtimes.avg() or math.inf)
      print( f"{W:4d}, {H:4d}: {Rtimes:8.3f} ms    {JtJtimes:8.3f} ms "
             f"  {JtJtimes/Rtimes:8.3f}" )
      RESULTS.append({ "problem":"img_ARAP",
                       "prob_rettype":"vector",
                       "compiler":compiler_str,
                       "N":W*H,
                       "mask_number":mask_no,
                       "f_ms_avg":Rtimes,
                       "JtJ_ms_avg":JtJtimes,
                     })

  def ATL():
    from perf_ATL.imgARAP import get_time
    do_run("ATL",get_time)
  ATL()

  def JAX():
    from perf_jax.imgARAP import get_time
    do_run("JAX",get_time)
  JAX()

img_ARAP()


RESULTS = pd.DataFrame(RESULTS)
RESULTS.to_csv(str(os.path.join(_HERE_DIR,"results.csv")))

print(RESULTS)



