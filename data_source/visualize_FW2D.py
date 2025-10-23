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
from read_FW2d import FW2DReader

#%%
class FW2DVisualizer:
    """
    A comprehensive visualization and analysis tool for 2D full-wave simulations.
    Provides electric field plots, density maps, Doppler spectra, and time evolution.
    """

    def __init__(self, subdir, f0, angle, mode, simulation, **kwargs):
        self.simname = definitions.get_simname(f0, angle, mode, simulation, **kwargs)
        self.input = InputData.load_pickle(self.simname, subdir=subdir, **kwargs)
        self.output = OutputData(self.simname, subdir=subdir, **kwargs)

        self.header = f"f0 = {int(f0*1e-9)} MHz — θ = {angle}° — {mode} mode"
        self.title = f"f0{int(f0*1e-9)}_θ{angle}_{mode}mode"

        self.f0 = f0
        self.angle = angle
        self.mode = mode
        self.simulation = simulation
        self.subdir = subdir

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
        self.data = FW2DReader.get_doppler_spectra(
            self.subdir, self.f0, self.angle, self.mode,
            self.simulation, simname=self.simname, **kwargs
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(self.data.freqs / self.data.fD,
                10 * np.log10(self.data.Raw_signal / self.data.Signal.max()),
                "r", label="Raw")
        ax.plot(self.data.Freqs / self.data.fD,
                10 * np.log10(self.data.Signal / self.data.Signal.max()),
                "g", label="Processed")
        ax.axvline(self.data.fD_Sim, c="k", ls="--", lw=0.8,
                   label=rf"$f/f_D$ = {self.data.fD_Sim:.3f}$")

        if show_fit:
            ax.plot(self.data.Freqs / self.data.fD,
                    10 * np.log10(self.data.gaussian_fit / self.data.Signal.max()),
                    "fuchsia", lw=2, label="Gaussian Fit")
            ax.plot(self.data.Freqs / self.data.fD,
                    10 * np.log10(self.data.lorentian_fit / self.data.Signal.max()),
                    "dodgerblue", lw=3, label="Lorentzian Fit")

        ax.set_title(f"{fit_type} {self.header}", fontsize=14)
        ax.set_ylim(-40, 10)
        ax.set_xlabel(r"$f/f_D$")
        ax.set_ylabel("PSD [a.u.]")
        ax.grid(ls="-.", c="silver", lw=0.5)
        ax.legend(loc="upper right")

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
