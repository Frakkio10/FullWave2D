#%% 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.constants as cnst
from DBS.beamtracing import DIFDOP
from DBS.io.utils import is_iterable
from matlabtools import Struct
from DBS import definitions as defs
from pathlib import Path
from fullwave2d.core.wrapper import InputData, OutputData
from fullwave2d.config.definitions import DATA_BEAM_DIR, LPP_palette
from config.definitions import HD5_DIR
import h5py

#%%

def compute_deltak(output):
    idxrw, idxpw = output.beam[0].inrayrw - 1, output.beam[0].inrayapol - 1
    return np.array([np.mean(np.abs(output.dif[i].k[idxrw, idxpw] - output.dif[i].k[0, 0])) for i in range(output.beam.size)])
    
from dataclasses import dataclass

unitfactordict = {'m': 1e-2, 'cm': 1, 'mm': 10}
VALID_VIEWS = ("poloidal", "toroidal")


@dataclass
class BeamPlotOptions:
    """Default plotting options for beam plots."""
    title: str = ""
    profi: int = 0
    waist: int = 0
    poloidal: int = 0
    toroidal: int = 0
    fig3d: int = 0
    txtanalytique: int = 0
    txtflux: int = 0
    txtrayons: int = 0
    arretrayon: int = 0
    vectordif: int = 0
    isophase: int = 0
    
def plot_beam(
    beam, dif, beami, ax=None, view="poloidal",
    central_ray=True, other_rays=True, unit="m", **plot_kwargs
):
    
    if ax is None:
        fig, ax = plt.subplots()
    uf = unitfactordict[unit]
    plots = BeamPlotOptions(
        **{k: plot_kwargs.pop(k, v) for k, v in BeamPlotOptions().__dict__.items()}
    )
    # Reshape rays
    ray = beam.ray
    beam.ray = ray.reshape(ray.shape[0], beami.nbrayr, beami.nbraya, ray.shape[-1])
    icoordx, icoordy = (0, 2) if view == "poloidal" else (0, 1)
    # Central ray
    if central_ray:
        R = beam.ray[:, 0, 0, icoordx] * uf
        YZ = beam.ray[:, 0, 0, icoordy] * uf
        ax.plot(R , YZ, **plot_kwargs)
    # Other rays
    if other_rays and beami.nbrayr > 1:
        plot_other_rays(ax, beam, dif, beami, plots, view, uf, icoordx, icoordy, **plot_kwargs)
    # Diffraction vector
    if plots.vectordif:
        ax.plot(dif.x[0, 0] * uf, dif.z[0, 0] * uf, "xk")
        ax.quiver(
            dif.x[0, 0] * uf, dif.z[0, 0] * uf,
            dif.kx[0, 0] * uf, dif.kz[0, 0] * uf,
            angles="xy", scale_units="xy", scale=1, color="k"
        )
    
def plot_other_rays(ax, beam, dif, beami, plots, view, uf, icoordx, icoordy, **plot_kwargs):
    inrayr = np.arange(1, beami.nbrayr)
    for inra in beam.inrayapol - 1:
        inra = inra if view == "poloidal" else inra - 1
        for inrr in inrayr:
            tfin = dif.inr[inrr, inra] if plots.arretrayon else beam.ray.shape[0]
            R = beam.ray[:tfin, inrr, inra, icoordx] * uf
            Z = beam.ray[:tfin, inrr, inra, icoordy] * uf
            ax.plot(R , Z, alpha=0.6, **plot_kwargs)
            
            
class Beam3dInterfaceSlab():

    def __init__(self, launcher, densityprof, outpath=None, nrays_radial=1, nrays_azimuthal=1, rhomapcorr=0, **kwargs):

        self.launcher = launcher
        self.densityprof = densityprof
        self.outpath = outpath
        self.nrays_radial = nrays_radial
        self.nrays_azimuthal = nrays_azimuthal
        self.rhomapcorr = rhomapcorr

        self.setup_beam3d_input()
        
    def setup_beam3d_input(self):
        """Translates input objects passed to the Class constructor into a `beam3d_input    object which has the format expected by the beam3d code."""

        if is_iterable(self.launcher): # todo: generalize this to densityprof and equilibrium, although in practictie it is probably not needed (i.e., keeping the same input equilibirum and density profile, but varying the launcher frequency and/or angle only, around a given time)
            self.beam3d_input = []
            for l in self.launcher:
                self.beam3d_input.append(self._setup_single_beam3d_input(l))
        else:
            self.beam3d_input = self._setup_single_beam3d_input()

    def _setup_single_beam3d_input(self, launcher = None, densityprof = None):
        
        if launcher is None:
            launcher = self.launcher
        if densityprof is None:
            densityprof = self.densityprof
            
        beam3d_input = Struct()
        beam3d_input.beami = self._setup_beami(launcher)
        beam3d_input.plasma = self._setup_plasma(launcher, densityprof)
        #====
        beam3d_input.integopt = self._setup_integopt()
        # beam3d_input.integopt.nx2mapdx = nx2

        return beam3d_input
    
    def _setup_beami(self, launcher):
    
        beami = Struct(
            diagnm            = launcher.name,
            Rgola             = launcher.Rgola * 1e2,   # [cm] 
            zgola             = launcher.zgola * 1e2,   # [cm] 
            ygola             = launcher.ygola * 1.0e2, # [cm]
            phigoladeg        = 0.0,   #float(launcher.phigoladeg), # [deg]
            thetadirgolarefdeg= 180.0, #float(launcher.thetadirgolarefdeg), # [deg] 180
            waist             = launcher.waist * 1e3, # [mm],
            dwaist            = 0.0, # 0.1 , #launcher.dwaist * 1.0e2, # [cm]
            antenna           = launcher.antenna,

            # other beam inputs (unrelated to the specific launcher properties, but to how the beam is modeled):
            diffract    = 1, # 0: crossing rays, 1: diffracting beam
            nbrayr      = float(self.nrays_radial), # nombre de rayons radialement 1+8n [2 5  9 17]
            nbraya      = float(self.nrays_azimuthal), # nombre de rayons par tour 4m      [4 8 16 32]
            tok         = launcher.machine,
            modex       = launcher.modex,
            uhrdif      = 0, # resonance scattering

            frei              = float(launcher.freqGHz), 
            phidirgoladeg     = 0.0, #float(launcher.angletor), # orientation toroidale de l'antenne par rapport à la visee centrale [deg]
            thetadirgolareldeg= float(launcher.anglepol), 
            smax = 300.0
        )
        return beami 

    def _setup_plasma(self, launcher, densityprof, **kwargs):
        
        nc = get_ncrit(launcher.freqGHz *1e9, angle = launcher.anglepol)
        ic = np.argmin(np.abs(densityprof.ne - nc))
        m, q = np.polyfit(densityprof.rho_psi, densityprof.ne, 1)

        slabcenter = densityprof.rho_psi[ic] 
        slabgradl  = -q/m  - densityprof.rho_psi[ic] 
        
        plasma = Struct(
            tok = 'air',
            nx2type = 'slabnx2', 
            equitype = 'none',
            netype = 'data', 
            nx2prof_type = 'given',  # 'given' or 'withne'
            typeripple= 0, 
            rhobord = 1.3, 
            neprofilerho = densityprof.rho_psi, 
            neprofile   = densityprof.ne, 
            slabcenter = slabcenter , 
            slabgradl = slabgradl, 
            r0tok = 236.,
            a0tok = 100., 
            b0tok = 100., 
            R0_b = 236., 
        )
        
        # plasma = Struct(
        #     tok = 'air',
        #     nx2type = 'purexo',
        #     netype = 'map', 
        #     equitype = 'none',
        #     #netype = 'data', 
        #     # nx2prof_type = 'withne',  # 'given' or 'withne'
        #     typeripple= 0, 
        #     rhobord = 1.3, 
        #     nerz_r  = densityprof.rgrid,
        #     nerz_z  = densityprof.zgrid, 
        #     nerz  = densityprof.ne * 1e-19,
        #     r0tok = 236.,
        #     a0tok = 100., 
        #     b0tok = 100., 
        #     R0_b = 236., 
        #     rgrid = densityprof.rgrid,
        #     zgrid = densityprof.zgrid, 
        # )

        return plasma
    
    def _setup_integopt(self):
        integopt = Struct(
            testklim       = 1e-4,
            corrknorm      = 1,
            rhomapcorr     = 0, 
        )
        return integopt
    
    def run_beam3d(self, outpath=None, verbose=False):

        if outpath is None:
            outpath = self.outpath

        if outpath is None:
            outpath = str(Path('./beam3d_output.mat').resolve())
        
        self.outpath = outpath # update if necessary
        
        # save as temporary .mat file (just so we can run the matlab script):
        id = np.random.randint(1e8) # random id to ensure uniqueness (only needed temporarily)
        p = defs.DATA_TMP_DIR / f'beam3d_input/{id}.mat'
        print(p)
        # p = Path('/Home/FO278650/tmp/DBSdata/tmp/') / f'beam3d_input/{id}.mat'
        if not p.parent.exists():
            p.parent.mkdir(parents=True)

        # we want the to export a list of inputs to be processed in matlab (this will turn into a cell array in matlab)
        if is_iterable(self.beam3d_input):
            pass
        else:
            self.beam3d_input = [self.beam3d_input]

        import scipy.io as sio
        sio.savemat(p, {'beam3d_input': self.beam3d_input, 'outpath': str(outpath)})
        # runs the matlab script:
        import os
        # path to the matlab script:
        #current_folder = Path(__file__).parent
        #p_matlab = str(current_folder / 'src')
        p_matlab = str('/home/FO278650/Bureau/DBS/DBS-toolkit/DBS/beamtracing/src')
        p_logdir = defs.LOG_DIR / f'beamtracing'
        if not p_logdir.exists():
            p_logdir.mkdir(parents=True)

        # print('Launching beam3d... (this may take a while)')
        # print('Overview of input parameters:')
        # print('-----------------------------')
        # print(self.beam3d_input[0].beami)

        from matlabtools import run_matlab_script
        # success = 0
        
        cmd_prefix = 'matlab -nodesktop -nosplash -r'
        
        if 'partenaires.cea.fr' in defs.CURR_HOSTNAME or 'intra.cea.fr' in defs.CURR_HOSTNAME:
            cmd_prefix = 'module load tools_dc && ' + cmd_prefix
        if 'toki' in defs.CURR_HOSTNAME:
            cmd_prefix = 'module load matlab/R2024bU3 && ' + cmd_prefix
            
            
            # Extract everything from "/u/username" onward
            print(p_matlab)
            path = Path(p_matlab)
            u_index = path.parts.index("u")
            subpath = Path(*path.parts[u_index:])
            subpath = Path('/') / subpath
    
            p_matlab = str(subpath)
        
        print(p_matlab)
        print('next line is running')
        success = run_matlab_script(f'run_beam3d({id});', addpath=f'{p_matlab}', logdir=p_logdir, cmd_prefix=cmd_prefix, verbose=verbose)
        return success
    
    def fetch_result(self):
        """
        Convenience method to fetch the result from the pre-defined output path.
        """
        return Beam3dInterfaceSlab.fetch_result_from_outpath(self.outpath)
    
    @classmethod
    def fetch_result_from_outpath(cls, path):
        """
        Fetch the beamtracing (.mat) output file located at a fiven path. Can be called without initializing the class.
        """
        
        result =  Struct.from_mat(path, 'outp')
        # check if the result is a list of results (i.e., multiple frequencies). If not, we need to wrap it in a list for consistency:
        if type(result.beam) != np.ndarray:

            for key,val in result.items():
                if key in ['k_perp', 'rho', 'freqGHz', 'beam', 'dif', 'beami', 'integopt']:
                    result[key] = np.array([val])
                    
        return Struct.from_mat(path, 'outp')

def get_nprof():

    from scipy.interpolate import CubicSpline

    x = np.linspace(0, 1024*2e-4, 1024) 
    # ne_lin = -10 * (x - x.max()) * 1.2e19 + 3e19
    ne_lin = -25 * (x - x.max()) * 2.2e19 + 3e17
    return x, ne_lin 
    
    
def get_ncrit(f0, angle=0.0):
    """
    Critical density for vacuum-frequency f0 [Hz] for
    O/X-mode and incidence angle [degrees].
    """
    e, me, eps0 = cnst.e, cnst.m_e, cnst.epsilon_0      # C, kg, F/m
    # eps0 * m_e * (2 pi)² /e² [SI units] = 0.012404426
    prefactor = eps0 * me * (2 * np.pi)**2 / e ** 2 
    return prefactor * f0**2 * np.cos(np.deg2rad(angle))




#%%
from scipy import constants as cnst 
import h5py

mi = 6 * cnst.m_p #kg
Te = 1800 #eV
B = 1.0 #T
cs = np.sqrt(cnst.e * Te / mi)
OmegaI = cnst.e * B / mi
rhos = cs / OmegaI
print(rhos)

flname = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C0.05.h5'
it = 5
with h5py.File(flname, 'r', libver='latest', swmr=True) as fl:

    #Construct the real space grid
    Lx, Ly = fl['params/Lx'][()], fl['params/Ly'][()]
    Npx, Npy = fl['params/Npx'][()], fl['params/Npy'][()]
    Nx, Ny, Nxh, Nyh = int(Npx/3)*2, int(Npy/3)*2, int(Npx/3), int(Npy/3)
    X, Y = np.arange(0,Nx)*Lx/Nx, np.arange(0,Ny)*Ly/Ny 
    x, y = np.meshgrid(X, Y, indexing='ij')
    # Linear parameters
    C, kap = fl['params/C'][()], fl['params/kap'][()]
    nu, D = fl['params/nu'][()], fl['params/D'][()]
    dkx, dky = 2*np.pi/Lx, 2*np.pi/Ly
    Kx, Ky = np.r_[np.arange(0,int(Nx/2)+1)*dkx, np.arange(-int(Nx/2)+1, 0)*dkx],  np.arange(0, int(Ny/2)+1)*dky
    kx, ky = np.meshgrid(Kx, Ky, indexing='ij')

#%%


##################### HW maps ###############################

# from FW2D.io.interface import DataInterface

# subdir, machine  = 'HW_rad_scan_C0.2_10', 'irene'
# subdir, machine  = 'HW_kscan_bump_C0.2', 'irene'

outpath = DATA_BEAM_DIR.joinpath(f'test_C0.005_well.mat')
# os.makedirs(os.path.dirname(outpath), exist_ok=True)

# params    = DataInterface(subdir, machine = machine).params
# inp       = InputData.load_pickle(params.name[0], subdir = subdir, machine = machine)

# anglepol = params.theta
# freqGHz  = params.F
# xmode    = 0
# N        = anglepol.size
# anglepol = np.array([5, 10, 15, 20, 25, 30, 40])
# freqGHz  = np.array([58.2, 59, 60.4, 62.5, 65.5, 69.5, 81.5])
# anglepol = np.array([5, 10, 15, 20, 25, 30])
# freqGHz  = np.array([69, 70, 71.8, 74.5, 78.2, 83.2])
anglepol = np.array([5, 10, 15, 20, 25, 30])
freqGHz  = np.array([50.2, 50.8, 52, 53.8, 56, 59])

xmode    = 0
N        = anglepol.size


# anglepol = np.array([10, 15, 20, 25, 30])
# freqGHz  = np.array([43.0, 44.2, 46, 48.6, 52])
# xmode    = 0
# N        = anglepol.size

launcher  = [DIFDOP(freqGHz=freqGHz[i], anglepol=anglepol[i], modex=xmode) for i in range(N)]


# densityprof = Struct(
#     rho_psi = X[-1700:-200] * rhos,
#     ne    = - kap * (X[-1700:-200] - Lx ) * 30e17 + 3e17
# )

densityprof = Struct(
    rho_psi = X[-1600:-100] * rhos,
    ne    = - kap * (X[-1600:-100] - Lx ) * 9e17 + 1e12
)

# m, q = np.polyfit(densityprof.rho_psi, densityprof.ne, 1)

# for i in range(N):
#     inp       = InputData.load_pickle(params.name[i], subdir = subdir, machine = machine)
#     m = np.tan(np.deg2rad(inp.angle))
#     q = Y[-1600 + int(inp.ny / 2 + inp.yante / inp.dx)] * rhos - m * X[-1] * rhos

#     launcher[i].waist = params.waist[i]
#     launcher[i].Rgola = 0.6 #densityprof.rho_psi.max() #- q / m
#     launcher[i].zgola = m * 0.60 + q
#     launcher[i].ygola = 0.0

for i in range(N):
    # inp       = InputData.load_pickle(params.name[i], subdir = subdir, machine = machine)
    m = np.tan(np.deg2rad(anglepol[i]))
    q = Y[-1600 + 250] * rhos - m * X[-1] * rhos

    launcher[i].waist = 0.05962969953486939
    launcher[i].Rgola = 1.6 #densityprof.rho_psi.max() #- q / m
    launcher[i].zgola = m * 1.60 + q
    launcher[i].ygola = 0.0


interface = Beam3dInterfaceSlab(launcher, densityprof, outpath = outpath, nrays_radial=1, nrays_azimuthal=1)
interface.run_beam3d(outpath = outpath)

#%%%

##################### ST maps ###############################

from scipy import constants as cnst 
import h5py

rhos = 2.2e-3


flname = '/home/FO278650/Bureau/FullWave2D_FO/data/maps/full_maps/mixed_advection_3.h5'
with h5py.File(flname, 'r', libver='latest', swmr=True) as fl:

    #Construct the real space grid
    x        = fl['grid/x'][:]
    y        = fl['grid/y'][:]
    
    
from FW2D.io.interface import DataInterface

subdir, machine  = 'mixed_advection_3_novphase', 'irene'
# subdir, machine  = 'HW_kscan_bump_C0.2', 'irene'

outpath = DATA_BEAM_DIR.joinpath(f'{subdir}.mat')
os.makedirs(os.path.dirname(outpath), exist_ok=True)

params    = DataInterface(subdir, machine = machine).params
inp       = InputData.load_pickle(params.name[0], subdir = subdir, machine = machine)

anglepol = params.theta
freqGHz  = params.F
xmode    = 0
N        = anglepol.size


# anglepol = np.array([10, 15, 20, 25, 30])
# freqGHz  = np.array([43.0, 44.2, 46, 48.6, 52])
# xmode    = 0
# N        = anglepol.size

launcher  = [DIFDOP(freqGHz=freqGHz[i], anglepol=anglepol[i], modex=xmode) for i in range(N)]


# densityprof = Struct(
#     rho_psi = X[-1700:-200] * rhos,
#     ne    = - kap * (X[-1700:-200] - Lx ) * 30e17 + 3e17
# )

densityprof = Struct(
    rho_psi = x,
    ne      = -25 * (x - x.max()) * 2.2e19 + 3e17
)

# m, q = np.polyfit(densityprof.rho_psi, densityprof.ne, 1)

for i in range(N):
    inp       = InputData.load_pickle(params.name[i], subdir = subdir, machine = machine)
    m = np.tan(np.deg2rad(inp.angle))
    q = y[int(inp.ny / 2 + inp.yante / inp.dx)] - m * x[-1] 

    launcher[i].waist = params.waist[i]
    launcher[i].Rgola = 0.3 #densityprof.rho_psi.max() #- q / m
    launcher[i].zgola = m * 0.3 + q
    launcher[i].ygola = 0.0

interface = Beam3dInterfaceSlab(launcher, densityprof, outpath = outpath, nrays_radial=2, nrays_azimuthal=4)
interface.run_beam3d(outpath = outpath)



#%%
outp = interface.fetch_result()


#%%
plt.plot(X[-1600:-100], ubar[-1600:-100], c = 'k')
plt.axvline(outp.dif[0].x)

plt.show()

x = np.array([outp.dif[i].x for i in range(0, outp.freqGHz.size)])
plt.plot(x, outp.k_perp * 2, 'o')
plt.xlim(134, 144)
#%%
# beam = Struct.from_mat(DATA_BEAM_DIR.joinpath(f'HW_kscan_bump_C1.0.mat'), 'outp')
# x = np.array([beam.dif[i].x[0,0] for i in range(0, beam.freqGHz.size)])
# tht = np.array([beam.beami[i].thetadirgolareldeg for i in range(0, beam.freqGHz.size)])

plt.plot(put.dif.x, beam.k_perp, 'o')

# x = np.array([outp.dif[i].x for i in range(0, outp.freqGHz.size)])

# x = outp.dif.x
# plt.plot(x, outp.k_perp, 'X')
plt.axvline(42.13, c = 'k', alpha = 0.5, ls = '--')
plt.xlim(40, 45)



#%%
import matplotlib.colors as mcolors

idx = 6
inp       = InputData.load_pickle(params.name[idx], subdir = subdir, machine = machine)


fig, ax = plt.subplots()
out      = OutputData(params.name[idx], subdir = inp.subdir, machine = machine)
Ez = out.ez[int(inp.TFSF/2) : inp.ny + int(inp.TFSF/2), int(inp.TFSF/2) : inp.nx + int(inp.TFSF/2)] # (ny, nx) -> ok for pcolormesh

ax.pcolormesh(x , y , inp.ne, cmap = 'terrain')
plot_beam(outp.beam[idx], outp.dif[idx], outp.beami[idx], ax = ax, unit = 'm', c = 'k', lw = 3, other_rays = True)

Ez = np.flip(Ez, axis = 1) ** 2
E2_norm = Ez / np.percentile(Ez, 99) * 7.5  # scale so ~99th percentile hits ~7.5
E2_norm = np.clip(E2_norm, 0, 10)
noise_floor = 0.01  # tune this — try 0.02 to 0.1
E2_masked = np.where(E2_norm < noise_floor, np.nan, E2_norm)
levels = [1/np.e, 1.0, 2.5, 5.0, 7.5, 10.0]
norm   = mcolors.LogNorm(vmin=levels[0], vmax=levels[-1])

im = ax.contourf(x , y , E2_masked , cmap = 'jet', levels = levels, norm=norm)



#%%

fig, ax = plt.subplots(figsize = (5, 4))

ax.plot(X[-1500:] * rhos * 1e2, inp.ne.mean(axis = 0), c = LPP_palette[0], label = r'$n_e$ from FW')
ax.plot(densityprof.rho_psi * 1e2, densityprof.ne, c = 'r', label = 'input for b3d')
ax.axvline(launcher[0].Rgola * 1e2, c = 'k', ls = '--', label= 'Rgola')

nc = get_ncrit(54 *1e9, angle = 10)
ic = np.argmin(np.abs(densityprof.ne - nc))
ax.plot(densityprof.rho_psi[ic] * 1e2, densityprof.ne[ic], 'o' , c = LPP_palette[4])
ax.axvline(densityprof.rho_psi[ic] * 1e2, ls = '--', c = LPP_palette[4], label = 'slabcenter')
ax.grid(c = 'silver', ls = '--', lw = 0.5)
ax.axvspan(densityprof.rho_psi[ic] * 1e2, launcher[0].Rgola * 1e2, color = '#90B4FF', alpha=0.3, label = 'slabgradl')

ax.set_xlim(50, 54); ax.set_ylim(-0.5e17, 1.25e19)
ax.set_xlabel('x [cm]'); ax.set_ylabel(r'$n_e$'); ax.legend(loc = 3)
#%%
kp = 4 * np.pi * outp.freqGHz * 1e9 / cnst.c * np.sin(anglepol * np.pi / 180) / 100 

plt.plot(anglepol, kp, 'o')
plt.plot(anglepol, outp.k_perp * 2, 'x')

# %%
from FW2D.io.interface import DataInterface

subdir, machine  = 'refl_test_2', 'altair'
outpath = DATA_BEAM_DIR.joinpath(f'{subdir}.mat')
os.makedirs(os.path.dirname(outpath), exist_ok=True)

params    = DataInterface(subdir, machine = machine).params
inp       = InputData.load_pickle(params.name[0], subdir = subdir, machine = machine)
x, y,     = np.linspace(0, inp.nx * inp.dx, inp.nx), np.linspace(0, inp.ny * inp.dx, inp.ny)

anglepol = params.theta
freqGHz  = params.F
xmode    = 0
N        = anglepol.size

launcher  = [DIFDOP(freqGHz=freqGHz[i], anglepol=anglepol[i], modex=xmode) for i in range(N)]
densityprof = Struct(
    rho_psi = x,
    ne    =  -50 * (x - x.max()) * 1.6e19 + 3e17
)
m, q = np.polyfit(densityprof.rho_psi, densityprof.ne, 1)
for i in range(N):
    inp       = InputData.load_pickle(params.name[i], subdir = subdir, machine = machine)
    launcher[i].waist = params.waist[i]
    launcher[i].Rgola = -q/m #densityprof.rho_psi.max()
    launcher[i].zgola = y[-int(inp.ny / 2 - inp.yante / inp.dx)] 
    launcher[i].ygola = 0.0

interface = Beam3dInterfaceSlab(launcher, densityprof, outpath = outpath, nrays_radial=2, nrays_azimuthal=4)
interface.run_beam3d(outpath = outpath)

# %%

subdir, machine  = 'HW_rad_scan_map2', 'altair'
fig, ax = plt.subplots(figsize = (5, 4))

dI = DataInterface(subdir, machine = machine)

ax.plot(densityprof.rho_psi, densityprof.ne, c = 'tab:orange')

xc = []
for f0, theta, col in zip(dI.params.F, dI.params.theta, LPP_palette):
    nc = get_ncrit(f0 *1e9, angle = theta)
    ic = np.argmin(np.abs(densityprof.ne - nc))
    xc.append(densityprof.rho_psi[ic])
    ax.plot(densityprof.rho_psi[ic], densityprof.ne[ic], 'o', c= col, label = r'$f_0$ = %.1f GHz' %f0)

xc = np.array(xc)
ax.legend(ncol = 2, loc = 1); ax.set_xlabel('x [m]'); ax.set_ylabel(r'$n_e [m^{-3}]$ ')

ax.grid(c = 'silver', ls = '--', lw = 0.5)
# %%
from scipy import constants as cnst
kp = 4 * np.pi * dI.params.F * 1e9 / cnst.c * np.sin(dI.params.theta * np.pi / 180) / 100 # cm-1

fig, ax = plt.subplots(figsize = (5, 3))

# delta_k = compute_deltak(outp)
# delta_rho = compute_deltarho(outp)
ax.plot(xc, kp, 'Xr', markersize = 8, label = r'$(4\pi f_0 / c) \sin \theta$')
ax.plot(outp.rho, outp.k_perp * 2, 'o', c = 'b', label = 'b3d')

ax.legend(); ax.grid(c = 'silver', ls = '--', lw = 0.5); ax.set_xlabel(r'$x [m]$'); ax.set_ylabel(r'$k_\perp [cm^{-1}]$')
# %%
##########################################################
###########  simul gauss #################
##########################################################

from scipy import constants as cnst 
import h5py

mi = 6 * cnst.m_p #kg
Te = 1800 #eV
B = 1.0 #T
cs = np.sqrt(cnst.e * Te / mi)
OmegaI = cnst.e * B / mi
rhos = cs / OmegaI
print(rhos)

flname = '/home/FO278650/Zone_Travail/HWAK/simu_hwak/4096_4096_C1.0.h5'
it = 8
with h5py.File(flname, 'r', libver='latest', swmr=True) as fl:

    #Construct the real space grid
    Lx, Ly = fl['params/Lx'][()], fl['params/Ly'][()]
    Npx, Npy = fl['params/Npx'][()], fl['params/Npy'][()]
    Nx, Ny, Nxh, Nyh = int(Npx/3)*2, int(Npy/3)*2, int(Npx/3), int(Npy/3)
    X, Y = np.arange(0,Nx)*Lx/Nx, np.arange(0,Ny)*Ly/Ny 
    x, y = np.meshgrid(X, Y, indexing='ij')
    # Linear parameters
    C, kap = fl['params/C'][()], fl['params/kap'][()]
    nu, D = fl['params/nu'][()], fl['params/D'][()]


from FW2D.io.interface import DataInterface

subdir, machine  = 'HW_kscan_bump_C0.2', 'irene'

outpath = DATA_BEAM_DIR.joinpath(f'simul_gaus_f49.mat')
os.makedirs(os.path.dirname(outpath), exist_ok=True)

params    = DataInterface(subdir, machine = machine).params
inp       = InputData.load_pickle(params.name[3], subdir = subdir, machine = machine)

anglepol = params.theta[3]
freqGHz  = params.F[3]
xmode    = 0
N        = anglepol.size

launcher  = DIFDOP(freqGHz=freqGHz, anglepol=anglepol, modex=xmode)
densityprof = Struct(
    rho_psi = X[-1500:] * rhos,
    ne    = - kap * (X[-1500:] - Lx ) * 30e17 + 3e17
)

inp       = InputData.load_pickle(params.name[3], subdir = subdir, machine = machine)
m = np.tan(np.deg2rad(inp.angle))
q = Y[-1500 + int(inp.ny / 2 + inp.yante / inp.dx)] * rhos - m * X[-1] * rhos

launcher.waist = params.waist[3]
launcher.Rgola = 0.6 #densityprof.rho_psi.max() #- q / m
launcher.zgola = m * 0.60 + q
launcher.ygola = 0.0


interface = Beam3dInterfaceSlab(launcher, densityprof, outpath = outpath, nrays_radial=17, nrays_azimuthal=32)
interface.run_beam3d(outpath = outpath)
# %%
################### before simu 

fig, ax = plt.subplots(figsize = (5, 2))

ax.plot(X[-1600:-100] * rhos * 1e2, ubar[-1600:-100], c = 'k')
xc = []
icl = []
# for f0, col in zip([60,59,58,57,56,55,54], LPP_palette):
for f0, col in zip([52], LPP_palette):
    nc = get_ncrit(f0 *1e9, angle = np.abs(10))
    ic = np.argmin(np.abs(densityprof.ne - nc))
    # print(X[-1500 + ic] * 1e2 * rhos, ic)
    ax.plot(X[-1700 + ic] * rhos * 1e2, ubar[-1700 + ic], 'o', c = col, label = r'$f_0$ = %d GHz' %f0)


    
ax.set_ylim(-1.5, 1.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)
ax.set_xlabel('x [cm]')
ax.set_ylabel(r'$v_{ZF}$')
ax.legend(loc = 2, ncol = 3, fontsize = 8)

xc = np.array(xc)
icl = np.array(icl)
# %%
fig, ax = plt.subplots()
axu = ax.twinx()

axu.plot(X[-1600:-100], ubar[-1600:-100], lw = 2, c = 'k')

with h5py.File(flname, 'r', libver='latest', swmr=True) as fl:
    nk = fl['fields/density/nk'][200]
    n = np.fft.irfft2(nk, norm='forward')
        
ax.pcolormesh(X[-1600:-100], Y[-1600:-100], n[-1600:-100,-1600:-100], cmap = 'seismic')

plot_beam(outp.beam[1], outp.dif[1], outp.beami[1], ax = ax, c = 'k', lw = 3, unit = 'cm')
# %%
