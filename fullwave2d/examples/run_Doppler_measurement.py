
#%%
from fullwave2d.main.mpi_maxwell import scatterv_maxwell
from fullwave2d.main.turbulence_map import SynthTurbMap, lin_prof
from fullwave2d.core.wrapper import InputData
import time
from mpi4py import MPI
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

simulations_per_CPU = 1

root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root

save_diag = True if is_root else False

# global simulation parameters
f0 = 60e9 # probing frequency [Hz]
# nt = 250
nt = 4200 # number of time steps
ny,nx = 700,700
lambda0 = scipy.constants.c/f0
# dx = lambda0 / 15 # spatial resolution [m]
dx = 1.5e-4
print(f'lambda0 / dx = {lambda0 / dx} (recommended > 20)')

waist = 30 * dx
yante = (float(ny)/2-300) * dx
angle = 15.
rms = 0.02 # intensity of fluctuations as a fraction of the (local) background density
name = 'test_doppler_2'

dny = 2
ysteps = size * simulations_per_CPU
ny_tot = ny+ysteps*dny

lambda_perp = lambda0 / 2 / np.sin(angle * np.pi / 180)

if is_root:
    assert(dny * dx < lambda_perp / 2)

ky = 4.5e2  # [rad/m]
kx = 5.e2  # [rad/m]

def _set_up_turb():
    # Set up homogeneous 2d Gaussian turbulence map (slightly anisotropic)
    ly = 1./ky  # m
    lx = 1./kx  # m
    beta = 0 * pi / 180  # rad 0 means no inclination of eddies
    map_args = (dx, nx, ny_tot, lx, ly, beta)
    turbObj = SynthTurbMap(map_args)
    Gauss_turbulence = turbObj.delta_ne
    return Gauss_turbulence


if rank == root:
    # define a place holder for the density map:
    ne = np.zeros((ny_tot, nx)).astype(np.double)
    # NOTE: The axes (x,y) of ne are reversed, because the source script
    # maxwell_2d_omode.c expects a 2d array of shape (ny, nx)

    # intitalize with linear background profile and
    # cutoff at a fraction 'cut' of the horizontal extent in the plasma
    lin_prof(ne, f0, cut=0.7, angle=angle)
    ne = np.flip(ne, axis=1)

    # add fluctuations
    delta_ne = np.flip(_set_up_turb(), axis=1)

    ne *= (1 + delta_ne * rms)
    ne = np.array(ne)

else:
    ne = None

# input to the full wave script
inp = InputData(
    name=name,
    f0=f0,
    nt=nt,
    nx=nx,
    ny=ny,
    dx=dx,
    ne=None,  # will be assigned in scatterv_maxwell()
    waist=waist,
    angle=angle,
    yante=yante,
    save_diag=save_diag
)
#%%
# perform calculation (using mpiexec (--use-hwthread-cpus) -n <n> python -m mpi4py run_Doppler_measurement.py)
if not size==1:
    # time the computation
    t0 = time.time()

    # lauch parallelized full-wave simulations
    outp_gathered = scatterv_maxwell(ne, dny, ysteps, inp, root=root)

    # save the results
    if rank == root:
        print(outp_gathered.shape)
        print('time (s): ', time.time() - t0)
        np.save(inp.get_outp_dir() / 'ampl_phase.npy', outp_gathered)

#%%
# plot the results (using simple native python or IPython)
if size==1:
    from fullwave2d.visualize import show_input_turbulence_spec
    fig, ax = plt.subplots()
    show_input_turbulence_spec(delta_ne, dx, ax1=ax)
    # compare with the prescriped k values:
    for k,l in zip([ky, kx], ax.get_lines()):
        ax.axvline(k / 100, ls='--', c=l.get_color())
    
    
    
    # launch a single test full wave simulation (not in parallel)
    # t0 = time.time()
    # inp.ne = ne
    # fw2d_wrapper(inp)
    # print('time (s) : ', time.time() - t0)
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from fullwave2d.core.wrapper import OutputData
    outp = OutputData('test_doppler')
    outp.display_results()
    
    # fig, ax = plt.subplots()

    from scipy.signal import welch
    ampl, phase = outp.doppler_data[:,0], outp.doppler_data[:,1]
    z = ampl * np.exp(1j * phase)
    f,Pxx = welch(z, return_onesided=False)
    
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    
    fig, ax = plt.subplots()
    ax.plot(f, 10 * np.log10(Pxx))
    # ax.plot(outp.phase)
    
    dy = dny * dx
    dt = 1 # arbitrary for the moment
    vperp = dy / dt
    kperp = 4*np.pi*f0 / scipy.constants.c *  np.sin(angle * np.pi / 180)
    fdoppler = kperp * vperp / 2 / pi
    ax.axvline(fdoppler / 2, color='k', ls='--')


# %%
