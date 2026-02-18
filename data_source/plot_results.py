#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as C, pi as PI

from fullwave2d.core.wrapper import InputData, OutputData
from fullwave2d import definitions
from data_source._read_FW2d import FW2DReader
from matlabtools import Struct
# %%
FWS_advection = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection.mat')
FWS_advection_ITG = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_ITG.mat')
FWS_advection_TEM = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_TEM.mat')
FWS_advection_mixed = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_mixed.mat')



# %%
%matplotlib inline
Nx, Ny, dx = 1024, 1024, 2e-4
dky = 2*np.pi / (Ny * dx)
ky = dky * np.r_[0:int(Ny/2 + 1)]
ky_c       = 0.8e3
omega0     = 5
omega_ITG = omega0 * ky / (1 + (ky / ky_c)**2)
omega_TEM = -omega0 * ky / (1 + (ky / ky_c)**2)
ky_c      = 0.8e3
omega0    = -5
omega_TEM2 = omega0 * ky / (1 + (ky / ky_c)**2)
#%matplotlib inline
fig, ax = plt.subplots(figsize = (6,3))
U0 = -10
ax.plot(FWS_advection.kp, FWS_advection.v_perp  , 'or', label = 'Advection')
# ax.plot(FWS_advection_ITG.kp, FWS_advection_ITG.v_perp  , 'ob', label = 'Advection + ITG')
# ax.plot(FWS_advection_TEM.kp, FWS_advection_TEM.v_perp  , 'og', label = 'Advection + TEM')
# ax.plot(FWS_advection_mixed.kp, FWS_advection_mixed.v_perp  , 'om', label = 'ITG + TEM')
# ax.plot(FWS_advection_mixed2.kp[0], -(2 * np.pi * 441) / (FWS_advection_mixed2.kp[0] * 1e2), 'Xm')
# ax.plot(FWS_advection_mixed2.kp[0], -(2 * np.pi * 192) / (FWS_advection_mixed2.kp[0] * 1e2), 'Xm')
# ax.plot(FWS_advection_mixed2.kp[1], -(2 * np.pi * 740) / (FWS_advection_mixed2.kp[1] * 1e2), 'Xm')
# ax.plot(FWS_advection_mixed2.kp[1], -(2 * np.pi * 1240) / (FWS_advection_mixed2.kp[1] * 1e2), 'Xm')
# ax.plot(FWS_advection_mixed2.kp[2], -(2 * np.pi * 1390) / (FWS_advection_mixed2.kp[2] * 1e2), 'Xm')
# ax.plot(FWS_advection_mixed2.kp[2], -(2 * np.pi * 1911) / (FWS_advection_mixed2.kp[2] * 1e2), 'Xm')    
# ax.plot(FWS_advection_mixed2.kp[3], -(2 * np.pi * 2310) / (FWS_advection_mixed2.kp[3] * 1e2), 'Xm')
# ax.plot(FWS_advection_mixed2.kp[3], -(2 * np.pi * 2789) / (FWS_advection_mixed2.kp[3] * 1e2), 'Xm')
ax.plot(ky*1e-2, U0 * np.ones_like(ky) , '--r', alpha = 0.6)
ax.plot(ky*1e-2, (U0 + omega_ITG / ky ), '--b', alpha = 0.6)
# ax.plot(ky*1e-2, (U0 + omega_TEM2 / ky ),  '--g', alpha = 0.6)
# ax.plot(ky*1e-2, (U0 + (omega_ITG + omega_TEM2) / ky ),  '--m', alpha = 0.6)

ax.legend()
# ax.set_ylim(0,2)
# ax.axhline(-10, ls = '--', c = 'silver')
# ax.set_ylim(-16,-4)
ax.grid(c = 'silver', ls = '--', lw = 0.5)
ax.set_xlim(0, 20)
ax.set_xlabel(r'$k_\perp$ [$cm^{-1}$]', fontsize = 10)
ax.set_ylabel(r'$v_\perp / U_0$ ', fontsize = 10)
# %%
plt.close()
%matplotlib widget
fig, ax = plt.subplots(figsize = (5, 4))
fig, ax1 = plt.subplots(figsize = (5, 4))

subdir = 'FWS_advection_tilt_neg'
simname = 'advection_tilt_neg_f60_angle5_RC_waist300'


spectra = FW2DReader.get_doppler_spectra(
        subdir, simname = simname, welch_spectra=False,
        save_results=False, load_if_existing=True,
        machine='irene'
    )
ax.plot(spectra.Freqs * 1e-3,
                10 * np.log10(spectra.Signal / spectra.Signal.max()),
                "g")
spectra = FW2DReader.get_doppler_spectra(
        subdir, simname = simname, welch_spectra=True,
        save_results=False, load_if_existing=True,
        machine='irene'
    )
ax1.plot(spectra.Freqs * 1e-3,
                10 * np.log10(spectra.Signal / spectra.Signal.max()),
                "g")

subdir = 'FWS_advection_tilt_pos'
simname = 'advection_tilt_pos_f60_angle5_RC_waist300'


spectra = FW2DReader.get_doppler_spectra(
        subdir, simname = simname, welch_spectra=False,
        save_results=False, load_if_existing=True,
        machine='irene'
    )
ax.plot(spectra.Freqs * 1e-3,
                10 * np.log10(spectra.Signal / spectra.Signal.max()),
                "r")
spectra = FW2DReader.get_doppler_spectra(
        subdir, simname = simname, welch_spectra=True,
        save_results=False, load_if_existing=True,
        machine='irene'
    )
ax1.plot(spectra.Freqs * 1e-3,
                10 * np.log10(spectra.Signal / spectra.Signal.max()),
                "r")
plt.show()

# %%
