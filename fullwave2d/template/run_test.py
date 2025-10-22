#%%
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData
from fullwave2d import definitions
import time
import h5py as h5
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
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
filename = Path('/home/forlacchio/FlowWaveSim/output/h5/tilt/advection_only.h5')

with h5.File(filename, "r") as f:
    n = f["fields/n"][:]
    x = f["grid/x"][:]
    y = f["grid/y"][:]

X, Y = np.meshgrid(x, y, indexing="xy")
Nt, ny, nx = n.shape
dx = x[1]
delta_ne = n[100]

ne = np.zeros([ny, nx])
#linear density profile with the cutoff at pos%
pos_cutoff = 0.08
n_cutoff   = 4e19

for i in range(0, nx):
    ne[:, i] = n_cutoff * i * dx / pos_cutoff

ne = np.flip(ne, axis = 1)

fig, ax = plt.subplots(1, 2, figsize = (8, 3))

im1 = ax[0].pcolormesh(x, y, ne, cmap = 'Blues')
plt.colorbar(im1, ax = ax[0])

im2 = ax[1].pcolormesh(x, y, delta_ne, cmap = 'seismic')
plt.colorbar(im2, ax = ax[1])

plt.show()
ne_map = ne * (1 + 0.05 * delta_ne) 
ne_map = ne_map.astype(np.double)
plt.pcolormesh(x, y, ne_map, cmap = 'terrain')
plt.colorbar()
# %%
angle = 15
dx = x[1]
waist     = 30 * dx
nx, ny = ne_map.shape

inp = InputData(
    header    = '%.1f fluctuation levels -- O mode -- f0 = 60 GHz -- %d deg -- waist %d dx' %(angle, angle, int(waist / dx)),
    name      = f'advection_only',  
    subdir    = 'test_FWS',
    f0        = 60e9,
    nt        = 8000,
    nx        = nx,
    ny        = ny,
    dx        = dx,
    ne        = ne_map,  
    waist     = waist,
    angle     = angle,
    yante     = int(ny / 2 - 250) * dx,
    save_diag = save_diag, 
    mode      = 'O', 
    dny       = 5
)

fig, ax = plt.subplots(figsize = (4, 4))

ax.pcolormesh(x, y, inp.ne, cmap = 'terrain', alpha = 1)

#%%
t0 = time.time()

fw2d_wrapper(inp)
print('time (s) : ', time.time() - t0)
# %%
name      = inp.name
outp      = OutputData(inp.name, subdir = inp.subdir)

fig, ax = plt.subplots(figsize = (4, 4))

Ez = outp.ez[int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2), int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2)]

ax.pcolormesh(x, y, np.flip(Ez, axis = 1), cmap = 'jet')


ax.pcolormesh(x, y, inp.ne, cmap = 'terrain', alpha = 0.3)


fig, ax = plt.subplots(2, 1, figsize = (10, 6), sharex = True)
ax[0].set_title(r'Amplitude of $E_z$')
ax[0].plot(outp.ampl, c = 'dodgerblue')
ax[1].plot(outp.phase / np.pi, c = 'dodgerblue')
ax[0].set_ylabel('Amplitude [a.u.]')
ax[1].set_ylabel('Phase [$\pi$]')
ax[1].set_xlabel('Time [10 $\Delta t$]')
plt.subplots_adjust(wspace = 0 )

# %%
np.load('/home/forlacchio/FullWave2D_FO/data/fw2d/test_FWS/advection_only/ant_signal_t.npy')
# %%
