#%%
from fullwave2d.main.mpi_maxwell import scatterv_maxwell_from_h5
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData
from fullwave2d import definitions
import time
from mpi4py import MPI
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as C
from matplotlib import colors
import pickle
import h5py as h5
from config.definitions import HD5_DIR
from pathlib import Path
#%% 

# Set up the parallelization 

simulations_per_CPU = 1
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

#%%
#define the inputs parameters

mode            = 'O'
f0, nx, ny, dx  = 60e9, 1024, 1024, 2e-4
b0              = 2.5
waist, yante    = 150 * dx, int(ny / 2 - 250) * dx
angle           = 20
nt              = int(8000)
B0              = (b0 * np.ones([nx, ny])).astype(np.double)

subdir          = 'refl_test_2'
name            = f'refl_test_noturb_angle{angle}_waist{int(waist / dx)}'


ne_lin = np.zeros((nx, ny))
x, y = np.linspace(0, nx * dx, nx), np.linspace(0, ny * dx, ny)


for i in range(0, ny):
    ne_lin[:,i] = -50 * (x - x.max()) * 1.6e19 + 3e17


if size == 1:
    plt.pcolormesh(x, y, ne_lin.T  , cmap = 'terrain')
    plt.colorbar()
# %%
inp = InputData(
    header    = f'reflection test for beamtracing',
    name      = name,    
    subdir    = subdir,
    f0        = f0,
    nt        = nt,
    nx        = nx,
    ny        = ny,
    dx        = dx,
    ne        = None,  
    waist     = waist,
    angle     = angle,
    yante     = yante,
    save_diag = save_diag, 
    mode      = mode, 
    b0        = B0,
)
# %%
if not size==1:

    t0 = time.time()
    # outp_gathered = scatterv_maxwell_v2(ne_map, dn, dny, ysteps, inp, root=root)
    # outp_gathered = scatterv_maxwell_from_h5(inp, ne_lin, filename, t_start=0, t_end=None, root=0, fluct_lvl=0.02, simulations_per_CPU=simulations_per_CPU)
    # save the results
    # if rank == root:
    #     print(outp_gathered.shape)
    #     print('time (s): ', time.time() - t0)
    #     np.save(inp.get_outp_dir() / 'ampl_phase.npy', outp_gathered)

else:
    t0 = time.time()
    inp.ne = ne_lin.T.astype(np.double)
    fw2d_wrapper(inp)
    print('time (s) : ', time.time() - t0)
