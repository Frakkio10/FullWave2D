#%%
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData
from fullwave2d import definitions
import time
from mpi4py import MPI
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as C

#%% 
# Set up the parallelization 
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

#%%
# define the input parameters
mode           = 'O'
f0, nx, ny, dx = 60e9, 1024, 1024, 2e-4
angle          = 0
nt             = 8000
yante          = int(ny / 2) * dx

subdir         = 'PCR_test'
name           = 'PCR_test_noturb_1tx_3rx'

# PCR receiver array: 1 transmitter, 3 receivers
y_center   = ny//2 * dx
delta_y    = 10 * dx          # spacing between receivers (m)
recv_width = 2                # half-width in grid points

n_recv = 3
yrecv  = np.array([
    ny//2 - 1,   # receiver left of center
    ny//2,       # receiver at center
    ny//2 + 1,   # receiver right of center
], dtype=np.int32) 
# in grid points, spacing = delta_y / dx = 10 cells between each

# simple linear density profile
ne_lin = np.zeros((nx, ny))
x = np.linspace(0, nx * dx, nx)
for i in range(ny):
    ne_lin[:, i] = -50 * (x - x.max()) * 1.6e19 + 3e17

if size == 1:
    plt.pcolormesh(ne_lin.T, cmap='terrain')
    plt.colorbar()
    plt.title('density profile')
    plt.show()

#%%
inp = InputData(
    name         = name,
    subdir       = subdir,
    f0           = f0,
    nt           = nt,
    nx           = nx,
    ny           = ny,
    dx           = dx,
    ne           = None,
    angle        = angle,
    yante        = yante,
    save_diag    = save_diag,
    mode         = mode,
    antenna_type = 'horn',
    horn_width   = 50 * dx,    # modest aperture for first test
    horn_length  = 0.5,        # m
    n_recv       = n_recv,
    yrecv        = yrecv,
    recv_width   = recv_width,
)

# correct yrecv to account for TFSF offset
y_center_grid = ny//2 + inp.TFSF
yrecv = np.round(
    np.arange(n_recv) * (delta_y / dx) + y_center_grid - (n_recv//2) * (delta_y/dx)
).astype(np.int32)
inp.yrecv = yrecv  # update the input object

#%%
if not size == 1:
    pass  # MPI to be added later
else:
    t0 = time.time()
    inp.ne = ne_lin.T.astype(np.double)
    fw2d_wrapper(inp)
    print('time (s) : ', time.time() - t0)
# %%
if size == 1:
    inp       = InputData.load_pickle(name, subdir = subdir)
    outp      = OutputData(inp.name, subdir = inp.subdir)

    fig, ax = plt.subplots(figsize = (4, 4))

    Ez = outp.ez[int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2), int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2)] # (ny, nx) -> ok for pcolormesh 

    ax.pcolormesh(np.flip(Ez, axis = 1), cmap = 'jet')
# %%
    # reconstruct complex signal per receiver:
    S0 = outp.recv_IQ[:, 0] + 1j * outp.recv_IQ[:, 1]  # receiver 0
    S1 = outp.recv_IQ[:, 2] + 1j * outp.recv_IQ[:, 3]  # receiver 1
    S2 = outp.recv_IQ[:, 4] + 1j * outp.recv_IQ[:, 5]  # receiver 2
    
    
    plt.plot(outp.ampl, label = 'outp.ampl')
    plt.plot(S0, label = 'S0')
    plt.plot(S1, label = 'S1')
    plt.plot(S2, label = 'S2')

    plt.legend()
# %%
