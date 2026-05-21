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
from scipy import constants as cnst 
mi = 6 * cnst.m_p #kg
Te = 1800 #eV
B = 1.0 #T
cs = np.sqrt(cnst.e * Te / mi)
OmegaI = cnst.e * B / mi
rhos = cs / OmegaI
print(rhos)

#%%

def get_ncrit_X(f0, b0, angle=0.0):
    """
    Compute cutoff densities nR, nL for X-mode according to the cutoff
    frequencies as found e.g. here:
    https://en.wikipedia.org/wiki/Electromagnetic_electron_wave

    Args:
        - f0 (float): vaccum-frequency in [Hz]
        - b0 (float): magnetic field in [Tesla]
        - angle (degrees): incidence angle # NOTE: is a factor of cos(angle) correct for X-mode as well?
    Returns:
        - (nR, nL) (float): lower and upper cut-off densities
    """
    from scipy.constants import m_e, e, epsilon_0

    # freq in rad/s:
    om = 2*np.pi* f0

    # cyclotron freq
    omc = e * b0 / m_e

    _R = (2*om - omc)**2 - omc**2
    _L = (2*om + omc)**2 - omc**2

    prefac = m_e * epsilon_0 / 4 / e**2
    angle_fac = np.cos(np.deg2rad(angle))

    nR = prefac * _R * angle_fac
    nL = prefac * _L * angle_fac

    return nR, nL

# Set up the parallelization 

simulations_per_CPU = 1
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

# %%
it = 11


# filename = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C1.0_v2.h5'
filename = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C1.0.h5'

with h5.File(filename, "r",  libver='latest', swmr=True) as f:
    t = f['fields/t'][()] #timestep at which the map is taken
    Lx, Ly = f['params/Lx'][()], f['params/Ly'][()]
    Npx, Npy = f['params/Npx'][()], f['params/Npy'][()]
    C, kap = f['params/C'][()], f['params/kap'][()]
    nu, D = f['params/nu'][()], f['params/D'][()]
    # nk = f['fields/uk'][it, 1]
    nk = f['fields/density/nk'][it]
f.close()

n = np.fft.irfft2(nk, norm='forward')
# delta_ne = n[652:1676]
# delta_ne = n[-1024:,-1024:]
delta_ne = n[-1500:, -1500:]

Nx, Ny, Nxh, Nyh = int(Npx/3)*2, int(Npy/3)*2, int(Npx/3), int(Npy/3)
X, Y = np.arange(0,Nx)*Lx/Nx, np.arange(0,Ny)*Ly/Ny 
x, y = np.meshgrid(X, Y, indexing='ij')
# ne_lin = - kap * (x[-1024:,-1024:] - Lx ) * rhos * 40e19 + 3e17
# ne_lin = -kap * (x - Lx) * 2.7e18 + 3e17
ne_lin = - kap * (x[-1500:, -1500:] - Lx ) * 30e17 + 3e17
#%%
x *= rhos
y *= rhos
dx =  x[1,0]
x = x[-1500:, -1500:]
y = y[-1500:, -1500:]

#%%
#define the inputs parameters

mode            = 'X'
f0, nx, ny, dx  = 70e9, delta_ne.shape[0], delta_ne.shape[1], dx
waist, yante    = 300 * dx, int(ny / 2 - 500) * dx
angle           = 10
t_step          = 1
nt              = int(10e3)
# B0              = (b0 * np.ones([nx, ny])).astype(np.double)
b0 = 1.5      # Tesla
L_grad = 1.25   # meters (realistic scale length)
# Magnetic field map
B0 = b0 * (1 - x  / L_grad).astype(np.double)


subdir          = 'HW_rad_scan_map2_X'
name            =  f'HW_test_f{int(f0*1e-9)}_angle{angle}_waist{int(waist / dx)}_long'
#%%

ne_tot = ne_lin* (1 + 0.3 * delta_ne / delta_ne.max())
if size == 1:
    fig, ax = plt.subplots(figsize = (5, 4))
    axi = ax.twinx()
    nR, nL = get_ncrit_X(f0, 1.0, angle=10.0)
    print('%.1e %.2e %.2e' %(f0, nR, nL))
    nc = nR if nR > 0 else nL
    nr = np.mean(ne_tot, axis = 1)
    print(nc)
    ic = np.argmin(np.abs(nr - nc))

    # _x = X[652:1676]
    _x = X[-1500:]
    _y = Y[-1500:]
    im = ax.pcolormesh(_x * rhos  * 1e2, _x * rhos  * 1e2, ne_tot.T  , cmap = 'terrain')
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
    outp_gathered = scatterv_maxwell_HW(inp, ne_lin, filename, t_start=11, t_end=None, root=0, fluct_lvl=0.2, simulations_per_CPU = simulations_per_CPU, t_step = t_step)
    # save the results
    if rank == root:
        print(outp_gathered.shape)
        print('time (s): ', time.time() - t0)
        np.save(inp.get_outp_dir() / 'ampl_phase.npy', outp_gathered)
    print('time (s): ', time.time() - t0)

else:
    t0 = time.time()
    inp.ne = ne_tot.T.astype(np.double)
    fw2d_wrapper(inp)
    print('time (s) : ', time.time() - t0)

# %%
