#%%
from data_source._read_FW2d import FW2DReader
import matplotlib.pyplot as plt 
import numpy as np
from matlabtools import Struct
#%%
FWS_advection = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection.mat')

angle, i = 5,0
subdir = 'FWS_advection_mixed'
simname = f'advection_mixed_f60_angle{angle}_RC_waist300'
fig, ax = plt.subplots(figsize = (6,3))
%matplotlib inline

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "m", label="mixed")
# ax.plot(spectra.Freqs * 1e-3,
#         10 * np.log10(spectra.lorentian_fit / spectra.Signal.max()),
#         "dodgerblue", lw=5, label="Lorentzian Fit")

subdir = 'FWS_advection_ITG'
simname = f'advection_ITG_f60_angle{angle}_RC_waist300'
spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "b", label="ITG")

subdir = 'FWS_advection_TEM'
simname = f'advection_TEM_f60_angle{angle}_RC_waist300'
spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "g", label="TEM")

ax.legend()
ax.set_title(r'Frequency spectra from FW2D -- $\theta$ = %.d°' %angle)
ax.set_xlabel(r'$f - f_D^{^{advection}}$ [kHz]', fontsize = 12)
ax.set_ylabel('PSD [a.u.]', fontsize = 14)
ax.set_xlim(-1.5,1.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)
#%%
angle, i = 40, 3
subdir = 'FWS_advection_mixed2'
simname = f'advection_mixed_f60_angle{angle}_RC_waist300'
fig, ax = plt.subplots(figsize = (6,3))
# %matplotlib inline

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "m", label="mixed 2")

ax.legend()
ax.set_title(r'Frequency spectra from FW2D -- $\theta$ = %.d°' %angle)
ax.set_xlabel(r'$f - f_D^{^{advection}}$ [kHz]', fontsize = 12)
ax.set_ylabel('PSD [a.u.]', fontsize = 14)
ax.set_xlim(-0.5,0.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)
# %%
angle, i = 40, 3
fig, ax = plt.subplots(figsize = (6,3))

# ax.plot(spectra.Freqs * 1e-3,
#         10 * np.log10(spectra.lorentian_fit / spectra.Signal.max()),
#         "dodgerblue", lw=5, label="Lorentzian Fit")

subdir = 'FWS_advection_ITG_2'
simname = f'advection_ITG_2_f60_angle{angle}_RC_waist300'
spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "b", label="ITG2", alpha = 0.6)

subdir = 'FWS_advection_TEM_3'
simname = f'advection_TEM_3_f60_angle{angle}_RC_waist300'
spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i])* 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "g", label="TEM3", alpha = 0.6)

subdir = 'FWS_advection_mixed_3'
simname = f'advection_mixed_3_f60_angle{angle}_RC_waist300'
%matplotlib inline

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "m", label="mixed3")

ax.legend()
ax.set_title(r'Frequency spectra from FW2D -- $\theta$ = %.d°' %angle)
ax.set_xlabel(r'$f - f_D^{^{advection}}$ [kHz]', fontsize = 12)
ax.set_ylabel('PSD [a.u.]', fontsize = 14)
ax.set_xlim(-1.5,1.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)

#%%
angle, i = 40, 3
subdir = 'mixed_advection'
simname = f'mixed_advection_f60_angle{angle}_RC_waist300'
fig, ax = plt.subplots(figsize = (6,3))
# %matplotlib inline

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "m", label="mixed 4")

ax.legend()
ax.set_title(r'Frequency spectra from FW2D -- $\theta$ = %.d°' %angle)
ax.set_xlabel(r'$f - f_D^{^{advection}}$ [kHz]', fontsize = 12)
ax.set_ylabel('PSD [a.u.]', fontsize = 14)
ax.set_xlim(-1.5,1.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)
# %%
# FWS_advection_tilt_pos = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_tilt_pos.mat')
# FWS_advection_tilt_neg = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_tilt_neg.mat')

angle, i = 25, 2
subdir = 'FWS_advection_tilt_pos'
simname = f'advection_tilt_pos_f60_angle{angle}_RC_waist300'
fig, ax = plt.subplots(figsize = (6,3))
%matplotlib inline

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs  - FWS_advection.fDop[i])* 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "r", label="tilt pos")


subdir = 'FWS_advection_tilt_neg'
simname = f'advection_tilt_neg_f60_angle{angle}_RC_waist300'
%matplotlib inline

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs  - FWS_advection.fDop[i]) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "b", label="tilt neg")


ax.legend()
ax.set_title(r'Frequency spectra from FW2D -- $\theta$ = %.d°' %angle)
ax.set_xlabel(r'$f - f_D^{^{advection}}$ [kHz]', fontsize = 12)
ax.set_ylabel('PSD [a.u.]', fontsize = 14)
ax.set_xlim(-1.5,1.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)

# %%
FWS_mixed_advection_2 = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_mixed_advection_2.mat')

angle, i = 40,5
subdir = 'mixed_advection_2'
simname = f'mixed_advection_2_f60_angle{angle}_RC_waist300'
fig, ax = plt.subplots(figsize = (5,3))

spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
ax.plot((spectra.freqs ) * 1e-3,
        10 * np.log10(spectra.Raw_signal / spectra.Signal.max()),
        "m", label="mixed")


ax.legend()
ax.set_title(r'Frequency spectra from FW2D -- $\theta$ = %.d°' %angle)
ax.set_xlabel(r'$f - f_D^{^{advection}}$ [kHz]', fontsize = 12)
ax.set_ylabel('PSD [a.u.]', fontsize = 14)
# ax.set_xlim(-1.5,1.5)
ax.grid(c = 'silver', ls = '--', lw = 0.5)
# %%
from fullwave2d.core.wrapper import InputData


inp = InputData.load_pickle(simname, subdir, machine = 'irene')
# %%
