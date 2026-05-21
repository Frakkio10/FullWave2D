#%%
from fullwave2d.main.mpi_maxwell import scatterv_maxwell_HW
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
# from config_FWS.definitions import HD5_DIR
from pathlib import Path
#%% 

def get_ncrit(f0, angle=0.0):
    """
    Critical density for vacuum-frequency f0 [Hz] for
    O/X-mode and incidence angle [degrees].
    """
    # eps0 * m_e * (2 pi)² /e² [SI units] = 0.012404426
    return f0**2 * 0.012404426 * np.cos(np.deg2rad(angle))

# Set up the parallelization 

simulations_per_CPU = 1
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

# %%
rhos = 10e-3
it = 10


filename = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C1.0_v2.h5'
with h5.File(filename, "r",  libver='latest', swmr=True) as f:
    t = f['fields/t'][()] #timestep at which the map is taken
    Lx, Ly = f['params/Lx'][()], f['params/Ly'][()]
    Npx, Npy = f['params/Npx'][()], f['params/Npy'][()]
    C, kap = f['params/C'][()], f['params/kap'][()]
    nu, D = f['params/nu'][()], f['params/D'][()]
    nk = f['fields/uk'][it, 1]
f.close()

n = np.fft.irfft2(nk, norm='forward')
# delta_ne = n[652:1676]
delta_ne = n[-1024:,-1024:]

Nx, Ny, Nxh, Nyh = int(Npx/3)*2, int(Npy/3)*2, int(Npx/3), int(Npy/3)
X, Y = np.arange(0,Nx)*Lx/Nx, np.arange(0,Ny)*Ly/Ny 
x, y = np.meshgrid(X, Y, indexing='ij')
ne_lin = - kap * (x[-1024:,-1024:] - Lx ) * rhos * 40e19 + 3e17

#%%
x *= rhos
y *= rhos
dx =  x[1,0]
x = x[-1024:,-1024:]
y = y[-1024:,-1024:]

#%%
#define the inputs parameters

mode            = 'O'
f0, nx, ny, dx  = 43e9, delta_ne.shape[0], delta_ne.shape[1], dx
b0              = 2.5
waist, yante    = 60 * dx, int(ny / 2 - 250) * dx
angle           = 10
t_step          = 1
nt              = int(6e3)
B0              = (b0 * np.ones([nx, ny])).astype(np.double)

subdir          = 'HW_rad_scan_10'
name            =  f'HW_test_f{int(f0*1e-9)}_angle{angle}_waist{int(waist / dx)}'


# ne_tot = ne_lin * (1 + 0.2 * n / n.max())
# ne_tot = ne_lin * (1 + 0.2 * delta_ne / delta_ne.max())
ne_tot = ne_lin * (1 + 0.2 * delta_ne / delta_ne.max())
if size == 1:
    fig, ax = plt.subplots(figsize = (5, 4))
    axi = ax.twinx()
    nc = get_ncrit(f0, angle = np.abs(angle))
    nr = np.mean(ne_tot, axis = 1)
    ic = np.argmin(np.abs(nr - nc))

    # _x = X[652:1676]
    _x = X[-1024:]
    im = ax.pcolormesh(x  * 1e2, y  * 1e2, ne_tot  , cmap = 'terrain')
    axi.plot(_x * rhos * 1e2, nr * 1e-19, c = 'w', lw = 5)
    axi.plot(_x * rhos * 1e2, nr * 1e-19, c = 'k', lw = 3)
    axi.plot(_x[ic] * rhos * 1e2, nr[ic] * 1e-19, 'Xw', markersize = 10)
    axi.plot(_x[ic] * rhos * 1e2, nr[ic] * 1e-19, 'Xr', markersize = 8)
    plt.colorbar(im, ax = ax)
    from scipy import constants as cnst
    lambda0  = cnst.c / f0 
    print(f'lambda0 / dx = {lambda0 / dx} (recommended > 20)')
    # plt.axis('equal')
# %%
inp = InputData(
    header    = f'HW map -- C = {C} -- filename {filename}',
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
    # print('in the mpi')
    t0 = time.time()
    outp_gathered = scatterv_maxwell_HW(inp, ne_lin, filename, t_start=10, t_end=None, root=0, fluct_lvl=0.2, simulations_per_CPU = simulations_per_CPU, t_step = t_step)
    # save the results
    if rank == root:
        print(outp_gathered.shape)
        print('time (s): ', time.time() - t0)
        np.save(inp.get_outp_dir() / 'ampl_phase.npy', outp_gathered)

else:
    t0 = time.time()
    inp.ne = ne_tot.T.astype(np.double)
    fw2d_wrapper(inp)
    print('time (s) : ', time.time() - t0)
