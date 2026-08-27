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
from matplotlib.patches import Rectangle
import pickle
import h5py as h5
from pathlib import Path

#%%
# ----------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------
from scipy import constants as cnst
mi      = 6 * cnst.m_p   # kg
Te      = 1800            # eV
B       = 1.0             # T
cs      = np.sqrt(cnst.e * Te / mi)
OmegaI  = cnst.e * B / mi
rhos    = cs / OmegaI
print(f'rhos = {rhos*100:.3f} cm')

def get_ncrit(f0, angle=0.0):
    return f0**2 * 0.012404426 * np.cos(np.deg2rad(angle))

#%%
# ----------------------------------------------------------------
# Parallelization setup
# ----------------------------------------------------------------
simulations_per_CPU = 4
root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root
save_diag = True if is_root else False

#%%
# ----------------------------------------------------------------
# Read HW turbulence file
# ----------------------------------------------------------------
it = 6
filename = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C0.2.h5'

with h5.File(filename, "r", libver='latest', swmr=True) as f:
    t       = f['fields/t'][()]
    Lx, Ly  = f['params/Lx'][()], f['params/Ly'][()]
    Npx, Npy = f['params/Npx'][()], f['params/Npy'][()]
    C, kap  = f['params/C'][()], f['params/kap'][()]
    nu, D   = f['params/nu'][()], f['params/D'][()]
    nk      = f['fields/density/nk'][it]

n        = np.fft.irfft2(nk, norm='forward')
delta_ne = n[-1600:-100, -1600:-100]

Nx, Ny   = int(Npx/3)*2, int(Npy/3)*2
X, Y     = np.arange(0, Nx)*Lx/Nx, np.arange(0, Ny)*Ly/Ny
x, y     = np.meshgrid(X, Y, indexing='ij')
ne_lin   = -kap * (x[-1600:-100, -1600:-100] - Lx) * 30e17 + 3e17

x  *= rhos
y  *= rhos
dx  = x[1, 0]
x   = x[-1600:-100, -1600:-100]
y   = y[-1600:-100, -1600:-100]

#%%
# ----------------------------------------------------------------
# Simulation parameters — PCR
# ----------------------------------------------------------------
mode   = 'O'
f0     = 50e9
nx, ny = delta_ne.shape[0], delta_ne.shape[1]
angle  = 0              # PCR: normal incidence
t_step = 1
nt     = int(8e3)

# transmitter centered on midplane
yante  = int(ny / 2) * dx

# ----------------------------------------------------------------
# PCR antenna array parameters (same philosophy as WEST setup)
# ----------------------------------------------------------------
n_recv         = 5
delta_y        = 180 * dx        # center-to-center spacing (m)
antenna_height = 0.8 * delta_y       # physical height of each receiver (m)
recv_width = int(antenna_height / 2 / dx)              # half-width of collection region (grid points)
horn_width     = antenna_height #(n_recv - 1) * delta_y * 1.5   # covers full array with margin
horn_length    = 0.07            # m

# ----------------------------------------------------------------
# Simulation name
# ----------------------------------------------------------------
subdir = 'HW_PCR_C0.2'
name   = (
    f'PCR_HW'
    f'_f{int(f0*1e-9)}GHz'
    f'_nrecv{n_recv}'
    f'_dy{int(delta_y/dx)}cells'
    f'_hw{int(horn_width/dx)}cells'
    f'_nt{nt}_2'
)

#%%
# ----------------------------------------------------------------
# Density profile check (single process only)
# ----------------------------------------------------------------
if size == 1:
    fig, ax = plt.subplots(figsize=(5, 4))
    axi = ax.twinx()
    nc  = get_ncrit(f0, angle=0)
    nr  = np.mean(ne_lin, axis=1)
    ic  = np.argmin(np.abs(nr - nc))
    im  = ax.pcolormesh(x[:, 0]*1e2, y[0, :]*1e2, ne_lin.T, cmap='terrain')
    axi.plot(x[:, 0]*1e2, nr*1e-19, c='w', lw=5)
    axi.plot(x[:, 0]*1e2, nr*1e-19, c='k', lw=3)
    axi.plot(x[ic, 0]*1e2, nr[ic]*1e-19, 'Xr', markersize=8)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title(r'$n_e$ background — HW C=1.0')
    lambda0 = cnst.c / f0
    print(f'lambda0 / dx = {lambda0/dx:.1f} (recommended > 20)')
    print(f'n_crit = {nc:.3e} m-3')
    print(f'n_max  = {ne_lin.max():.3e} m-3')
    plt.show()

#%%
# ----------------------------------------------------------------
# Build InputData
# ----------------------------------------------------------------
inp = InputData(
    header       = f'PCR HW map -- C={C} -- file {filename}',
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
    antenna_type = 'gaussian',   # wide Gaussian to illuminate all receivers
    waist        = horn_width,   # beam waist = horn_width for uniform illumination
    n_recv       = n_recv,
    yrecv        = np.zeros(n_recv, dtype=np.int32),  # placeholder, corrected below
    recv_width   = recv_width,
)
#%%

# ----------------------------------------------------------------
# Correct yrecv to C grid coordinates (accounts for TFSF offset)
# ----------------------------------------------------------------
y_center_grid = ny//2 + inp.TFSF
yrecv = np.round(
    np.arange(n_recv) * (delta_y/dx)
    + y_center_grid - (n_recv//2) * (delta_y/dx)
).astype(np.int32)
inp.yrecv = yrecv

if is_root:
    print(f'yrecv:          {yrecv}')
    print(f'y_center_grid:  {y_center_grid}')
    print(f'delta_y:        {delta_y*100:.2f} cm = {int(delta_y/dx)} cells')
    print(f'horn_width:     {horn_width*100:.2f} cm = {int(horn_width/dx)} cells')
    print(f'recv_width:     {recv_width} cells')
    print(f'no overlap:     {int(delta_y/dx)} > {2*recv_width} → {int(delta_y/dx) > 2*recv_width}')
    print(f'k_theta_max:    {pi/delta_y:.1f} m-1')
    print(f'k_theta_res:    {2*pi/(n_recv*delta_y):.1f} m-1')

#%%
# ----------------------------------------------------------------
# Visualize setup before running
# --------------------------------------------------------------#%%
# ----------------------------------------------------------------
# Visualize antenna positions on the density map
# ----------------------------------------------------------------
if size == 1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- left panel: full domain ---
    ax = axes[0]
    ax.pcolormesh(x[:, 0]*1e2, y[0, :]*1e2, ne_lin.T, cmap='terrain', alpha=0.8)
    ax.axvline(x=x[-1-inp.xante, 0]*1e2, color='gray', linestyle=':', linewidth=1, label='antenna plane')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title('Full domain')
    ax.legend()

    # --- right panel: zoom on antenna region ---
    ax = axes[1]
    ax.pcolormesh(x[:, 0]*1e2, y[0, :]*1e2, ne_lin.T, cmap='terrain', alpha=0.8)

    x_ant_phys = x[-1, 0] * 1e2   # cm
    rect_w     = 5 * dx * 1e2             # cm, cosmetic

    # transmitter (white)
    tx_y_phys  = y[0, ny//2] * 1e2
    tx_h_phys  = horn_width * 1e2
    ax.add_patch(Rectangle(
        (x_ant_phys - rect_w/2, tx_y_phys - tx_h_phys/2),
        width=rect_w, height=tx_h_phys,
        linewidth=2, edgecolor='white', facecolor='white', alpha=0.6, zorder=5))
    ax.text(x_ant_phys - 6*dx*1e2, tx_y_phys, 'TX',
            color='white', fontsize=9, va='center', ha='right', fontweight='bold')

    # receivers (cyan)
    yrecv_phys = y[0, inp.yrecv - inp.TFSF] * 1e2
    for r, yr in enumerate(yrecv_phys):
        ax.add_patch(Rectangle(
            (x_ant_phys - rect_w/2, yr - antenna_height*1e2/2),
            width=rect_w, height=antenna_height*1e2,
            linewidth=2, edgecolor='k', facecolor=LPP_palette[r], alpha= 1, zorder=5))
        ax.text(x_ant_phys - 6*dx*1e2, yr, f'RX{r}',
                color='cyan', fontsize=9, va='center', ha='right')

    # spacing annotation between RX0 and RX1
    ax.annotate('', xy=(x_ant_phys + 10*dx*1e2, yrecv_phys[1]),
                xytext=(x_ant_phys + 10*dx*1e2, yrecv_phys[0]),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
    ax.text(x_ant_phys + 12*dx*1e2, (yrecv_phys[0]+yrecv_phys[1])/2,
            f'Δy={delta_y*1e2:.1f} cm', color='k', fontsize=10, va='center')

    # zoom limits: ±30% beyond array span
    margin = (n_recv * delta_y * 1e2) * 0.3
    ax.set_ylim(yrecv_phys[0] - margin, yrecv_phys[-1] + margin)
    ax.set_xlim(x_ant_phys - 20*dx*1e2, x_ant_phys + 20*dx*1e2)
    ax.axvline(x=x_ant_phys, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title(f'Antenna plane zoom\n{n_recv} receivers, Δy={delta_y*1e2:.1f} cm, waist={horn_width*1e2:.1f} cm')

    plt.suptitle(f'PCR setup — HW C={C} — f0={f0*1e-9:.0f} GHz', fontsize=11)
    plt.tight_layout()
    plt.show()

    # --- print summary ---
    print(f'{"="*45}')
    print(f'PCR ANTENNA SETUP SUMMARY')
    print(f'{"="*45}')
    print(f'antenna plane:   x = {x_ant_phys:.2f} cm')
    print(f'TX center:       y = {tx_y_phys:.2f} cm')
    print(f'TX waist:        {horn_width*1e2:.2f} cm = {int(horn_width/dx)} cells')
    print(f'n_recv:          {n_recv}')
    print(f'delta_y:         {delta_y*1e2:.2f} cm = {int(delta_y/dx)} cells')
    print(f'recv_width:      {recv_width} cells = {recv_width*dx*1e2:.2f} cm')
    print(f'array span:      {(n_recv-1)*delta_y*1e2:.2f} cm')
    print(f'no overlap:      {int(delta_y/dx) > 2*recv_width}')
    print(f'k_theta_max:     {pi/delta_y:.1f} m-1')
    print(f'k_theta_res:     {2*pi/(n_recv*delta_y):.1f} m-1')
    print(f'yrecv (cells):   {inp.yrecv}')
    print(f'yrecv (cm):      {np.round(yrecv_phys, 2)}')
    print(f'{"="*45}')

#%%
# ----------------------------------------------------------------
# Run simulation
# ----------------------------------------------------------------
if not size == 1:
    t0 = time.time()
    outp_gathered = scatterv_maxwell_HW(
        inp, ne_lin, filename,
        t_start=10, t_end=None,
        root=0,
        fluct_lvl=0.2,
        simulations_per_CPU=simulations_per_CPU,
        t_step=t_step
    )
    if rank == root:
        print(f'outp_gathered shape: {outp_gathered.shape}')
        # columns: [amp_dbs, phase_dbs, ampl_r0, phase_r0, ..., ampl_r4, phase_r4]
        np.save(inp.get_outp_dir() / 'ampl_phase_pcr.npy', outp_gathered)
        print('time (s): ', time.time() - t0)

else:
    t0 = time.time()
    ne_tot   = ne_lin * (1 + 0.2 * delta_ne / delta_ne.max())
    inp.ne   = ne_tot.T.astype(np.double)
    fw2d_wrapper(inp)
    print('time (s): ', time.time() - t0)

#%%
# ----------------------------------------------------------------
# Load and plot single-process results
# ----------------------------------------------------------------
if size == 1:
    
    # name = 'PCR_HW_f54GHz_nrecv5_dy180cells_hw1080cells_nt10000'
    inp  = InputData.load_pickle(name, subdir=subdir)
    outp = OutputData(inp.name, subdir=inp.subdir)

    # E field
    fig, ax = plt.subplots(figsize=(5, 5))
    Ez = outp.ez[int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2),
                 int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2)]
    ax.pcolormesh(x[:, 0]*1e2, y[0, :]*1e2, np.flip(Ez, axis=1), cmap='jet')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title(name)
    plt.tight_layout()
    plt.show()

    # PCR receiver signals
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for r in range(inp.n_recv):
        axes[0].plot(outp.recv_ampl[:, r], label=f'RX{r}')
        axes[1].plot(outp.recv_phase[:, r], label=f'RX{r}')
    axes[0].set_ylabel('amplitude')
    axes[1].set_ylabel('phase (rad)')
    axes[1].set_xlabel('timestep')
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(f'PCR receiver signals — HW C={C}')
    plt.tight_layout()
    plt.show()

#%%
# ----------------------------------------------------------------
# Post-processing of MPI gathered results (root only)
# ----------------------------------------------------------------
if size > 1 and rank == root:

    data = np.load(inp.get_outp_dir() / 'ampl_phase_pcr.npy')
    # shape: (Nt_sel, 2 + 2*n_recv)

    amp_dbs   = data[:, 0]
    phase_dbs = data[:, 1]

    # complex signal per receiver per snapshot
    S = np.zeros((data.shape[0], n_recv), dtype=complex)
    for r in range(n_recv):
        S[:, r] = data[:, 2 + 2*r] * np.exp(1j * data[:, 3 + 2*r])

    # amplitude per receiver across snapshots
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in range(n_recv):
        ax.plot(np.abs(S[:, r]), label=f'RX{r}')
    ax.set_xlabel('snapshot')
    ax.set_ylabel('amplitude')
    ax.legend()
    ax.set_title('PCR amplitude per receiver — HW C=1.0')
    plt.tight_layout()
    plt.show()

    # cross-correlation matrix
    Gamma = np.zeros((n_recv, n_recv), dtype=complex)
    for r1 in range(n_recv):
        for r2 in range(n_recv):
            Gamma[r1, r2] = np.mean(S[:, r1] * np.conj(S[:, r2]))

    # k_theta spectrum
    gamma_1d = np.array([Gamma[0, r] for r in range(n_recv)])
    k_theta  = np.fft.fftfreq(n_recv, d=delta_y)
    S_ktheta = np.abs(np.fft.fft(gamma_1d))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.fft.fftshift(k_theta), np.fft.fftshift(S_ktheta))
    ax.set_xlabel(r'$k_\theta$ (m$^{-1}$)')
    ax.set_ylabel(r'$S(k_\theta)$')
    ax.set_title('PCR k_theta spectrum — HW C=1.0')
    plt.tight_layout()
    plt.show()

    print(f'k_theta_max: {pi/delta_y:.1f} m-1')
    print(f'k_theta_res: {2*pi/(n_recv*delta_y):.1f} m-1')
# %%
#%%    
    LPP_palette = ['#000090', '#90B4FF', '#FF5500', '#C00020', '#00A050', '#B40090', '#323232', '#E6E6E6', ]
    import matplotlib.colors as mcolors

    levels = [0.1, 1.0, 2.5, 5.0, 7.5, 8]
    levels = [1/np.e, 1.0, 2.5, 5.0, 7.5, 10.0]
    norm   = mcolors.LogNorm(vmin=levels[0], vmax=levels[-1])
    fig, ax = plt.subplots(1, 1, figsize=(5, 8),)
                        #  gridspec_kw={'height_ratios': [1, 0.4]}, sharex = True )
    # ax.pcolormesh(rgrid_fine, zgrid_fine, ne_2d, 
                # shading='auto', cmap='Blues', alpha=0.8)

    Ez = outp.ez[int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2), int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2)] # (ny, nx) -> ok for pcolormesh
    Ez = Ez ** 2
    E2_norm = Ez / np.percentile(Ez, 99) * 7.5  # scale so ~99th percentile hits ~7.5
    E2_norm = np.clip(E2_norm, 0, 10)
    noise_floor = 0.01  # tune this — try 0.02 to 0.1
    E2_masked = np.where(E2_norm < noise_floor, np.nan, E2_norm)
    im = ax.pcolormesh(X[-1600:-100] * rhos * 1e2 , Y[-1600:-100] * rhos * 1e2, inp.ne , cmap = 'terrain', shading = 'auto')
    ax.contourf(X[-1600:-100] * rhos * 1e2 , Y[-1600:-100] * rhos * 1e2, np.flip(E2_masked, axis = 1) , cmap = 'jet', levels = levels, norm=norm)
    ax.contour(X[-1600:-100] * rhos * 1e2 , Y[-1600:-100] * rhos * 1e2, np.flip(E2_masked, axis = 1), levels = levels, colors='k', linewidths=1, alpha=0.5)
    
    # transmitter rectangle (white)
    tx_y_phys  = y[ny//2]
    tx_h_phys  = horn_width


    # # receiver rectangles (cyan)
    yrecv_phys = y[0, inp.yrecv - inp.TFSF] * 1e2
    for r, yr in enumerate(yrecv_phys):
        ax.add_patch(Rectangle(
            (x_ant_phys - rect_w*5, yr - antenna_height*1e2),
            width=rect_w * 5, height=antenna_height*1e2,
            linewidth=1, facecolor=LPP_palette[r], edgecolor= 'k', alpha=1, zorder=5))

    # axs[1].plot(X[-1600:-100], ubar[-1600:-100], c = 'k', lw = 3)
    # axs[1].set_ylim(-3.5,3.5)
    ax.set_title(r'Beam propagation $f_0$=%d GHz $\theta$ =%d°' %(inp.f0 * 1e-9, -inp.angle), fontsize = 14)

    # ax.set_xlabel(r'$x [cm]$', fontsize = 14)
    # axs[1].set_ylabel(r'$v_{ZF}/ (\rho_s \Omega_i)$', fontsize = 14)
    # ax.set_ylabel(r'$y [cm]$', fontsize = 14)
    # axs[1].tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('R [m]', fontsize = 14); ax.set_ylabel('Z [m]', fontsize = 14)

    ax.set_aspect('equal')
    plt.subplots_adjust(hspace = 0.05)
    plt.show()
# %%
    LPP_palette = ['#000090', '#90B4FF', '#FF5500', '#C00020', '#00A050', '#B40090', '#323232', '#E6E6E6', ]

    fig, axes = plt.subplots(figsize=(6, 3), sharex=True)
    for r in range(inp.n_recv):
        axes.plot(outp.recv_ampl[:, r], label=f'RX{r}', lw = 2, c = LPP_palette[r])
    axes.set_ylabel('Amplitude [a.u.]')
    axes.set_xlabel('Time [10 $\Delta t$]')
    axes.legend(ncol = 2, fontsize = 10)
    axes.grid(c = 'silver', lw = 0.5, ls = '--')
    axes.set_title('PCR receiver signals')
    axes.tick_params(axis='both', labelsize=14)
# %%
