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
# ----------------------------------------------------------------
# Parallelization setup
# ----------------------------------------------------------------
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

#%%

# ----------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------
mode           = 'O'           # O-mode polarization
f0             = 60e9          # probing frequency (Hz)
nx, ny, dx     = 1024, 1024, 2e-4   # grid size and resolution (m)
angle          = 0             # PCR: normal incidence, no tilt
nt             = 5000          # number of time steps (increase for better IQ convergence)
yante          = int(ny/2) * dx  # transmitter centered at middle of grid

# ----------------------------------------------------------------
# PCR receiver array parameters
# ----------------------------------------------------------------
n_recv     = 3                 # number of receivers
delta_y    = 200 * dx          # physical spacing between receivers (m)
recv_width = 50                # half-width of each receiver collection region (grid points)
                               # no-overlap condition: delta_y/dx > 2*recv_width

# horn transmitter parameters
horn_width  = 500 * dx         # aperture width (m), should cover full receiver array
horn_length = 0.5              # horn length (m), controls phase curvature

# ----------------------------------------------------------------
# Informative simulation name
# ----------------------------------------------------------------
subdir = 'PCR_test'
name = (
    f'PCR'
    f'_{mode}mode'
    f'_f{int(f0*1e-9)}GHz'
    f'_nrecv{n_recv}'
    f'_dy{int(delta_y/dx)}cells'
    f'_hw{int(horn_width/dx)}cells'
    f'_nt{nt}'
    f'_noturb'
)

# ----------------------------------------------------------------
# Density profile (smooth linear profile, no turbulence)
# ----------------------------------------------------------------
ne_lin = np.zeros((nx, ny))
x = np.linspace(0, nx * dx, nx)
for i in range(ny):
    ne_lin[:, i] = -50 * (x - x.max()) * 1.6e19 + 3e17

if size == 1:
    plt.figure()
    plt.pcolormesh(ne_lin.T, cmap='terrain')
    plt.colorbar()
    plt.title('density profile')
    plt.show()
    

#%%
# ----------------------------------------------------------------
# Build InputData — note yrecv is set after inp is created
# because we need inp.TFSF to correctly place receivers in C grid
# ----------------------------------------------------------------
inp = InputData(
    name         = name,
    subdir       = subdir,
    f0           = f0,
    nt           = nt,
    nx           = nx,
    ny           = ny,
    dx           = dx,
    ne           = None,       # set later before fw2d_wrapper
    angle        = angle,
    yante        = yante,
    save_diag    = save_diag,
    mode         = mode,
    antenna_type = 'horn',     # horn for PCR, gaussian for DBS
    horn_width   = horn_width,
    horn_length  = horn_length,
    n_recv       = n_recv,
    yrecv        = np.zeros(n_recv, dtype=np.int32),  # placeholder, corrected below
    recv_width   = recv_width,
)

# ----------------------------------------------------------------
# Correct yrecv to account for TFSF offset in C grid
# receivers are evenly spaced and centered on the beam
# ----------------------------------------------------------------
y_center_grid = ny//2 + inp.TFSF
yrecv = np.round(
    np.arange(n_recv) * (delta_y/dx) + y_center_grid - (n_recv//2) * (delta_y/dx)
).astype(np.int32)
inp.yrecv = yrecv
print('yrecv:', yrecv)
print('y_center_grid:', y_center_grid)

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
    inp  = InputData.load_pickle(name, subdir=subdir)
    outp = OutputData(inp.name, subdir=inp.subdir)

    # --- E field snapshot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    Ez = outp.ez[int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2),
                 int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2)]
    ax.pcolormesh(np.flip(Ez, axis=1), cmap='jet')

    x_ant     = inp.nx - inp.xante
    rect_width = 6

    # transmitter rectangle (white): height = horn aperture
    tx_y      = ny//2
    tx_height = int(inp.horn_width / inp.dx)
    ax.add_patch(Rectangle(
        (x_ant - rect_width/2, tx_y - tx_height//2),
        width=rect_width, height=tx_height,
        linewidth=1.5, edgecolor='w', facecolor='w', alpha=0.5, zorder=5))
    ax.text(x_ant + rect_width, tx_y, 'TX', color='k', fontsize=7, va='center')

    # receiver rectangles (cyan): height = 2 * recv_width
    yrecv_plot = inp.yrecv - inp.TFSF
    for r, yr in enumerate(yrecv_plot):
        ax.add_patch(Rectangle(
            (x_ant - rect_width/2, yr - inp.recv_width),
            width=rect_width, height=2*inp.recv_width,
            linewidth=1.5, edgecolor='k', facecolor='k', alpha=0.4, zorder=5))
        ax.text(x_ant + rect_width, yr, f'RX{r}', color='k', fontsize=7, va='center')

    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    ax.set_title(name)
    plt.tight_layout()
    plt.show()

    # amplitude per receiver
    # complex signal per receiver, shape (n_timesteps, n_recv)
    S = outp.recv_S

    # or manually:
    S = outp.recv_ampl * np.exp(1j * outp.recv_phase)
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    # plot amplitude
    for r in range(inp.n_recv):
        axes[0].plot(outp.recv_ampl[:, r], label=f'RX{r}')

    # plot phase
    for r in range(inp.n_recv):
        axes[1].plot(outp.recv_phase[:, r], label=f'RX{r}')
        
    axes[1].set_ylabel('phase (rad)')
    axes[1].set_xlabel('timestep')
    axes[1].legend()

    plt.suptitle('PCR receiver signals')
    plt.tight_layout()
    plt.show()

    # --- cross-correlation between receivers (last timestep) ---
    S_final = S[-1, :]   # converged complex signal per receiver
    print('Amplitudes:', np.abs(S_final))
    print('Phases (rad):', np.angle(S_final))
    print('Phase differences (rad):', np.diff(np.angle(S_final)))
# %%
