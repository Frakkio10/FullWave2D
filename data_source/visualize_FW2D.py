#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.constants import c as C, pi as PI
from matplotlib.gridspec import GridSpec
from matplotlib import colors

from fullwave2d.core.wrapper import InputData, OutputData
from fullwave2d import definitions
from data_source._read_FW2d import FW2DReader

class FW2DVisualizer:
    """
    A comprehensive visualization and analysis tool for 2D full-wave simulations.
    Provides electric field plots, density maps, Doppler spectra, and time evolution.
    """

    def __init__(self, subdir, simname = None, **kwargs):
        if simname is None:
            f0         = kwargs.pop('f0', 60e9)
            angle      = kwargs.pop('angle', 15)
            mode       = kwargs.pop('mode', 'O')
            simulation = kwargs.pop('simulation', 'spectrum_study')
            simname = definitions.get_simname(f0, angle, mode, simulation, **kwargs)
        inp = InputData.load_pickle(simname, subdir=subdir, **kwargs)
        out = OutputData(simname, subdir=subdir, **kwargs)

        self.header = f"f0 = {int(inp.f0*1e-9)} GHz — θ = {-inp.angle}° — {inp.mode} mode"
        self.title = f"f0{int(inp.f0*1e-9)}_θ{-inp.angle}_{inp.mode}mode"
        self.subdir  = subdir
        self.simname = simname
        self.input   = inp
        self.output  = out

    # ─────────────────────────────────────────────
    # Field Plot
    # ─────────────────────────────────────────────
    def plot_field(self, show_density=False, **kwargs):
        fig, ax = kwargs.get("fig"), kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        Ez = self.output.ez[
            int(self.input.TFSF / 2): self.input.nx + int(self.input.TFSF / 2),
            int(self.input.TFSF / 2): self.input.ny + int(self.input.TFSF / 2),
        ]
        Ez = np.flip(Ez)
        self.Ez = Ez

        if show_density:
            ax.contour(Ez, colors="k", levels=[0.2, 0.4, 0.6, 0.8, 1],
                       norm=colors.Normalize(vmin=-1, vmax=1))
            self.plot_density_map(ax=ax, fig=fig, **kwargs)
        else:
            ax.imshow(Ez, cmap="jet", norm=colors.Normalize(vmin=-1, vmax=1))

        ax.set_title(self.header, fontsize=14)
        ax.axis("off")

        if kwargs.get("plot_evo", False):
            self.plot_time_evolution()

    # ─────────────────────────────────────────────
    # Density Map
    # ─────────────────────────────────────────────
    def plot_density_map(self, **kwargs):
        fig, ax = kwargs.get("fig"), kwargs.get("ax")
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        im = ax.imshow(self.input.ne * 1e-19, cmap="Blues", origin="lower")
        fig.colorbar(im, ax=ax, label=r"$n_e$ [$10^{19}$ m$^{-3}$]")

        if not kwargs.get("show_extent", False):
            ax.axis("off")
        else:
            ax.set_title("Density map", fontsize=14)

    # ─────────────────────────────────────────────
    # Time Evolution
    # ─────────────────────────────────────────────
    def plot_time_evolution(self, ax1=None, ax2=None, **plot_kwargs):
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

        ax1.plot(self.output.ampl, **plot_kwargs)
        ax1.set_ylabel("Amplitude [a.u.]", fontsize=10)

        ax2.plot(self.output.phase / np.pi, **plot_kwargs)
        ax2.set_ylabel(r"Phase [$\pi$]", fontsize=10)
        ax2.set_xlabel(r"10Δt", fontsize=10)

    # ─────────────────────────────────────────────
    # Doppler Spectra
    # ─────────────────────────────────────────────
    def plot_spectra(self, welch=False, show_fit=False, ax=None, **kwargs):
        fit_type = "Welch" if welch else "FFT"
        data = FW2DReader.get_doppler_spectra(self.subdir, simname=self.simname, **kwargs)
        col = kwargs.get('color', 'r')
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(data.freqs * 1e-3,
                10 * np.log10(data.Raw_signal / data.Signal.max()),
                "r", label="Raw")
        # ax.plot(data.freqs  * 1e-3,
        #         10 * np.log10(data.Raw_signal / data.Signal.max()),
        #          label="Raw", c = col)
        ax.plot(data.Freqs * 1e-3,
                10 * np.log10(data.Signal / data.Signal.max()),
                "g", label="Processed")
        # ax.axvline(data.fD_Sim * 1e-3, c="k", ls="--", lw=0.8,
        #            label=r"f = #%.1f kHz" #%(data.fD_Sim * 1e-3))

        if show_fit:
            #ax.plot(data.Freqs * 1e-3,
            #        10 * np.log10(data.gaussian_fit / data.Signal.max()),
            #        "fuchsia", lw=2, label="Gaussian Fit")
            ax.plot(data.Freqs * 1e-3,
                    10 * np.log10(data.lorentian_fit / data.Signal.max()),
                    "dodgerblue", lw=3, label="Lorentzian Fit")

        ax.set_title(f"{fit_type} {self.header}", fontsize=14)
        ax.set_ylim(-40, 10)
        ax.set_xlabel(r"$f [kHz]$")
        ax.set_ylabel("PSD [a.u.]")
        ax.grid(ls="-.", c="silver", lw=0.5)
        ax.legend(loc="upper right")
        self.data = data

    # ─────────────────────────────────────────────
    # Combined Plot
    # ─────────────────────────────────────────────
    def plot_all(self, save_fig=False, **kwargs):
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(4, 4, figure=fig)

        ax_field = fig.add_subplot(gs[:2, :2])
        self.plot_field(ax=ax_field)

        ax_amp = fig.add_subplot(gs[2, :2])
        ax_phase = fig.add_subplot(gs[3, :2])
        self.plot_time_evolution(ax1=ax_amp, ax2=ax_phase, c="b")

        ax_fft = fig.add_subplot(gs[:2, 2:])
        ax_welch = fig.add_subplot(gs[2:, 2:])
        self.plot_spectra(ax=ax_fft, welch=False, show_fit=True, **kwargs)
        self.plot_spectra(ax=ax_welch, welch=True, show_fit=True, **kwargs)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        if save_fig:
            file = definitions.get_figures_path(self.simname, self.subdir, **kwargs)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            plt.savefig(file)
            print(f"Saved figure: {file}")

        plt.show()

#%%
if __name__ == '__main__':
    from scipy import constants as cnst

    from matlabtools import Struct  
    beam = Struct.from_mat('/home/SR273822/data/DBSdata/processed/beamtracing/slab/simu_ref_gola_waist.mat', 'outp')

    subdir = 'mixed_advection_2'
    simname = 'mixed_advection_2_f60_angle40_RC_waist150_1024'
    # %matplotlib inline
    FW2d = FW2DVisualizer( subdir, simname = simname, machine = 'irene')
    FW2d.plot_field()
    FW2d.plot_time_evolution()
    
    plt.show()
    spectra = FW2DReader.get_doppler_spectra(
            subdir, simname = simname, welch_spectra=True,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
    print(spectra.fD_Lor,  spectra.fD)
    # %matplotlib widget
    fig, ax = plt.subplots(figsize = (5, 4))
    FW2d.plot_spectra(show_fit = True, machine = 'irene', welch_spectra = False, ax  = ax, rem_left = 5, rem_right = 5, color = 'r')
    plt.show()

    #rem_left = 30, rem_right = 40

#%%
    FWS_cyclonbase = Struct(
        theta = np.array([5, 10, 15, 20, 25, 30, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-288, -765,  -1132.8, -1439, -1679, -1992, -2656]),  #
        k_perp = beam.k_perp, 
    )
        
    FWS_cyclonbase.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_cyclonbase.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_cyclonbase.v_perp = (2 * np.pi * FWS_cyclonbase.fDop) / (FWS_cyclonbase.kp * 1e2)
    FWS_cyclonbase.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_cyclonbase.mat')
    
#%%
    FWS_mixed_advection_2 = Struct(
        theta = np.array([5, 10, 15, 20, 25, 30, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-338, -736,  -1097, -1445, -1778,-2070, -2652.9]),  #
        k_perp = beam.k_perp, 
    )
        
    FWS_mixed_advection_2.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_mixed_advection_2.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_mixed_advection_2.v_perp = (2 * np.pi * FWS_mixed_advection_2.fDop) / (FWS_mixed_advection_2.kp * 1e2)
    FWS_mixed_advection_2.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_mixed_advection_2.mat')
    #%%
    FWS_mixed_advection = Struct(
        theta = np.array([5, 10, 15, 20, 25, 30, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-321,-671,  -1055, -1444, -1790,-2120, -2673]), 
        # k_perp = beam.k_perp, 
    )
        
    FWS_mixed_advection.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_mixed_advection.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_mixed_advection.v_perp = (2 * np.pi * FWS_mixed_advection.fDop) / (FWS_mixed_advection.kp * 1e2)
    FWS_mixed_advection.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_mixed_advection.mat')
#%%
    FWS_advection_mixed_3 = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-332, -993, -1647, -2566]), #1073
        k_perp = beam.k_perp, 
    )
        
    FWS_advection_mixed_3.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_mixed_3.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_mixed_3.v_perp = (2 * np.pi * FWS_advection_mixed_3.fDop) / (FWS_advection_mixed_3.kp * 1e2)
    FWS_advection_mixed_3.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_mixed_3.mat')
#%%
    FWS_advection_TEM_3 = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-366.6, -1073, -1738, -2627]), #1073
        k_perp = beam.k_perp, 
    )
        
    FWS_advection_TEM_3.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_TEM_3.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_TEM_3.v_perp = (2 * np.pi * FWS_advection_TEM_3.fDop) / (FWS_advection_TEM_3.kp * 1e2)
    FWS_advection_TEM_3.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_TEM_3.mat')
#%%
    from matlabtools import Struct  
    
    FWS_advection_ITG_2 = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-312, -952, -1582, -2473]), 
        k_perp = beam.k_perp, 
    )
        
    FWS_advection_ITG_2.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_ITG_2.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_ITG_2.v_perp = (2 * np.pi * FWS_advection_ITG_2.fDop) / (FWS_advection_ITG_2.kp * 1e2)
    FWS_advection_ITG_2.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_ITG_2.mat')
#%%
    from matlabtools import Struct  
    
    FWS_advection_mixed2 = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-281, -936, -1567, -2572]), #976
        k_perp = beam.k_perp, 
    )
        
    FWS_advection_mixed2.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_mixed2.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_mixed2.v_perp = (2 * np.pi * FWS_advection_mixed2.fDop) / (FWS_advection_mixed2.kp * 1e2)
    FWS_advection_mixed2.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_mixed2.mat')

#%%

#%%
    from matlabtools import Struct  
    
    FWS_advection_mixed = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-358, -1030, -1676, -2609]), 
        k_perp = beam.k_perp, 
    )
        
    FWS_advection_mixed.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_mixed.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_mixed.v_perp = (2 * np.pi * FWS_advection_mixed.fDop) / (FWS_advection_mixed.kp * 1e2)
    FWS_advection_mixed.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_mixed.mat')
#%%
    from scipy import constants as cnst

    from matlabtools import Struct  
    beam = Struct.from_mat('/home/SR273822/data/DBSdata/processed/beamtracing/slab/simu_ref_gola_waist.mat', 'outp')


    # FWS_advection_shear_neg = Struct(
    #     theta = np.array([5, 15, 25, 40]),
    #     # fDop  = np.array([-422, -670, -1244]), 
    #     fDop  = np.array([-475, -1443, -2417, -3797]), 
    #     k_perp = beam.k_perp, 
    # )
    # FWS_advection_shear_neg.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_shear_neg.theta * np.pi / 180) / 100 # cm-1
    # # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    # FWS_advection_shear_neg.v_perp = (2 * np.pi * FWS_advection_shear_neg.fDop) / (FWS_advection_shear_neg.kp * 1e2)
    # FWS_advection_shear_neg.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_shear_neg.mat')
    
    FWS_advection_tilt_neg = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-391.7, -1024, -1718, -2545]), #335.6, -996.1, -1676, -2552
        k_perp = beam.k_perp, 
    )
    FWS_advection_tilt_neg.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_tilt_neg.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_tilt_neg.v_perp = (2 * np.pi * FWS_advection_tilt_neg.fDop) / (FWS_advection_tilt_neg.kp * 1e2)
    # FWS_advection_tilt_neg.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_tilt_neg.mat')
    
    
    FWS_advection_tilt_pos = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-351, -1024,  -1718, -2589]), #356 -1046,-1708, -2585
        k_perp = beam.k_perp, 
    )
    FWS_advection_tilt_pos.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_tilt_pos.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_tilt_pos.v_perp = (2 * np.pi * FWS_advection_tilt_pos.fDop) / (FWS_advection_tilt_pos.kp * 1e2)
    # FWS_advection_tilt_pos.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_tilt_pos.mat')
#%%
    from scipy import constants as cnst

    from matlabtools import Struct  
    beam = Struct.from_mat('/home/SR273822/data/DBSdata/processed/beamtracing/slab/simu_ref_gola_waist.mat', 'outp')

    
    FWS_advection = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-348.8, -1035.9, -1691, -2563 ]), 
        k_perp = beam.k_perp, 
    )
    FWS_advection.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.kp * 1e2)
    FWS_advection.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection.mat')
    
    FWS_advection_ITG = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-180.0, -703.1, -1384.8, -2320.9 ]), #220
        k_perp = beam.k_perp, 
    )
    FWS_advection_ITG.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_ITG.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_ITG.v_perp = (2 * np.pi * FWS_advection_ITG.fDop) / (FWS_advection_ITG.kp * 1e2)
    FWS_advection_ITG.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_ITG.mat')

    
    FWS_advection_TEM = Struct(
        theta = np.array([5, 15, 25, 40]),
        # fDop  = np.array([-422, -670, -1244]), 
        fDop  = np.array([-507.8, -1367.2, -2031.3, -2887.9 ]),  #2939
        k_perp = beam.k_perp, 
    )
    FWS_advection_TEM.kp = 4 * np.pi * 60e9 / cnst.c * np.sin(FWS_advection_TEM.theta * np.pi / 180) / 100 # cm-1
    # FWS_advection.v_perp = (2 * np.pi * FWS_advection.fDop) / (FWS_advection.k_perp[:-1] * 1e2)
    FWS_advection_TEM.v_perp = (2 * np.pi * FWS_advection_TEM.fDop) / (FWS_advection_TEM.kp * 1e2)
    FWS_advection_TEM.to_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_TEM.mat')

    
#%%
    from matlabtools import Struct
    plt.close()
    # ###%matplotlib inline 
    
    FWS_advection = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection.mat')
    FWS_advection_ITG_2 = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_ITG_2.mat')
    FWS_advection_TEM_3 = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_TEM_3.mat')
    FWS_advection_mixed = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_mixed.mat')
    Nx, Ny, dx = 1024, 1024, 2e-4
    dky = 2*np.pi / (Ny * dx)
    ky = dky * np.r_[0:int(Ny/2 + 1)]
    # ky_c       = 0.8e3
    # omega0     = 5
    ky_c       = 1.2e3
    omega0     = 1
    omega_ITG = omega0 * ky / (1 + (ky / ky_c)**2)
    omega_TEM = -omega0 * ky / (1 + (ky / ky_c)**2)

    ky_c      = 0.8e3
    omega0    = -3
    omega_TEM2 = omega0 * ky / (1 + (ky / ky_c)**2)
    U0 = - 10

    ####%matplotlib inline
    fig, ax = plt.subplots(figsize = (6,3))
    U0 = -10
    ax.plot(FWS_advection.kp, FWS_advection.v_perp  /U0, 'or', label = 'Advection')
    # ax.plot(FWS_advection_ITG.kp, FWS_advection_ITG.v_perp  /U0, 'ob', label = 'Advection + ITG')
    ax.plot(FWS_mixed_advection.kp, FWS_mixed_advection.v_perp  /U0, 'ob', label = 'Mixed')
    # ax.plot(FWS_advection_ITG_2.kp, FWS_advection_ITG_2.v_perp/U0  , 'ob', label = 'Advection + ITG 2')
    # ax.plot(FWS_advection_TEM_3.kp, FWS_advection_TEM_3.v_perp /U0 , 'og', label = 'Advection + mixed 3')
    # ax.plot(FWS_advection_mixed_3.kp, FWS_advection_mixed_3.v_perp /U0 , 'om', label = 'Advection + TEM 3')
    # ax.plot(FWS_advection_TEM.kp, FWS_advection_TEM.v_perp  , 'og', label = 'Advection + TEM')
    # ax.plot(FWS_advection_mixed.kp, FWS_advection_mixed.v_perp  , 'om')
    # ax.plot(FWS_advection_mixed2.kp[2], -(2 * np.pi * 1390) / (FWS_advection_mixed2.kp[2] * 1e2), 'Xb')
    # ax.plot(FWS_advection_mixed2.kp[2], -(2 * np.pi * 1995) / (FWS_advection_mixed2.kp[2] * 1e2), 'Xr')    
    # ax.plot(FWS_advection_mixed2.kp[1], -(2 * np.pi * 734) / (FWS_advection_mixed2.kp[1] * 1e2), 'Xb')
    # ax.plot(FWS_advection_mixed2.kp[1], -(2 * np.pi * 1296) / (FWS_advection_mixed2.kp[1] * 1e2), 'Xr')
    # ax.plot(FWS_advection_mixed2.kp[3], -(2 * np.pi * 2335) / (FWS_advection_mixed2.kp[3] * 1e2), 'Xb')
    # ax.plot(FWS_advection_mixed2.kp[3], -(2 * np.pi * 2837) / (FWS_advection_mixed2.kp[3] * 1e2), 'Xr')
    # ax.plot(FWS_advection_mixed2.kp[0], -(2 * np.pi * 504) / (FWS_advection_mixed2.kp[0] * 1e2), 'Xm')
    # ax.plot(ky*1e-2, U0 * np.ones_like(ky) /U0, '--r', alpha = 0.6)
    # ax.plot(ky*1e-2, (U0 + omega_ITG / ky ) /U0, '--b', alpha = 0.6)
    # ky_c       = 1.2e3
    # omega0     = -0.5
    # omega_TEM = omega0 * ky / (1 + (ky / ky_c)**2)
    # ax.plot(ky*1e-2, (U0 + omega_TEM / ky ) / U0, '--g', alpha = 0.6)
    # ax.plot(ky*1e-2, (U0 + (omega_ITG + omega_TEM) / ky ) / U0, '--m', alpha = 0.6)

    # ax.plot(ky*1e-2, (U0 + omega_TEM / ky ),  '--g', alpha = 0.6)
    # ax.plot(ky*1e-2, (U0 + (omega_ITG + omega_TEM) / ky ),  '--m', alpha = 0.6)
    
    
    ax.legend()
    ax.set_ylim(0.8,1.3)
    # ax.axhline(-10, ls = '--', c = 'silver')
    # ax.set_ylim(-11, -9)
    ax.grid(c = 'silver', ls = '--', lw = 0.5)
    ax.set_xlim(0, 20)
    ax.set_xlabel(r'$k_\perp$ [$cm^{-1}$]', fontsize = 10)
    ax.set_ylabel(r'$v_\perp / U_0$ ', fontsize = 10)
    
#%%
    # ###%matplotlib inline
    FWS_advection = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection.mat')
    # FWS_advection_tilt_neg = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_tilt_neg.mat')
    # FWS_advection_tilt_pos = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_tilt_pos.mat')
    fig, ax = plt.subplots(figsize = (6,3))
    U0 = -10
    ax.plot(FWS_advection.kp, FWS_advection.v_perp   , 'or', label = 'Advection')
    # ax.plot(FWS_advection_tilt_neg.kp, FWS_advection_tilt_neg.v_perp   , 'ob', label = 'Advection tilt neg')
    # ax.plot(FWS_advection_shear_neg.kp, FWS_advection_shear_neg.v_perp  , 'om', label = 'Advection shear neg')
    # ax.plot(FWS_advection_tilt_pos.kp, FWS_advection_tilt_pos.v_perp   , 'og', label = 'Advection tilt pos')
    ax.legend()
    # ax.set_ylim(0.95, 1.05)
    # ax.axhline(-10, ls = '--', c = 'silver')
    # ax.set_ylim(-11,-9)
    ax.grid(c = 'silver', ls = '--', lw = 0.5)
    ax.set_xlim(0, 20)
    ax.set_xlabel(r'$k_\perp$ [$cm^{-1}$]', fontsize = 10)
    ax.set_ylabel(r'$v_\perp / U_0$ ', fontsize = 10)
#%%

    # ###%matplotlib inline
    FWS_advection = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection.mat')
    FWS_advection_shear_neg = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/results/FWS_advection_shear_neg.mat')
    fig, ax = plt.subplots(figsize = (6,3))
    U0 = -10
    xcut = np.array([12, 12.8, 14, 16])
    x = np.linspace(0, 1024 * 2e-4, 1024)
    U0 = - 10 - 30 * (x)
    ax.plot(FWS_advection.kp, FWS_advection.v_perp   , 'or', label = 'Advection')
    ax.plot(FWS_advection_shear_neg.kp, FWS_advection_shear_neg.v_perp  , 'om', label = 'Advection shear neg')
    # ax.plot(x *1e2, U0, ls = '--', c = 'm')
    ax.legend()
    # ax.set_ylim(0.95, 1.05)
    # ax.axhline(-10, ls = '--', c = 'r')
    # ax.set_ylim(-11,-9)
    ax.grid(c = 'silver', ls = '--', lw = 0.5)
    ax.set_xlim(0, 20)
    ax.set_xlabel(r'$k_\perp$ [$cm^{-1}$]', fontsize = 10)

    # ax.set_xlabel(r'$x$ [cm]', fontsize = 10)
    ax.set_ylabel(r'$v_\perp$ ', fontsize = 10)

#%%
