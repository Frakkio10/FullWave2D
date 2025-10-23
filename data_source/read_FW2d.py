#%%
import os
import numpy as np
from scipy.constants import c as C, pi as PI
from scipy.signal import welch
from scipy.optimize import curve_fit
from fullwave2d.core.wrapper import InputData, OutputData
from fullwave2d import definitions
from matlabtools import Struct

#%%

def is_iterable(obj, allow_strings=False):
    """Check if an object is iterable (excluding dict by default)."""
    if isinstance(obj, dict):
        return False
    if isinstance(obj, str):
        return allow_strings
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class FW2DReader(Struct):
    """Retrieve simulation signal, compute Doppler spectrum, and perform Gaussian/Lorentzian fits."""

    def __init__(self, subdir, f0, angle, mode, simulation, **kwargs):
        simname = definitions.get_simname(f0, angle, mode, simulation, **kwargs)
        self.input = InputData.load_pickle(simname, subdir=subdir, **kwargs)
        self.output = OutputData(simname, subdir=subdir, **kwargs)

        self.f0 = f0
        self.angle = angle
        self.mode = mode
        self.subdir = subdir
        self.simname = simname

    @staticmethod
    def gaussian(x, amp, center, width):
        return amp * np.exp(-(x - center) ** 2 / (2 * width**2))

    @staticmethod
    def lorentzian(x, amp, center, width):
        return (amp / np.pi) * (width / 2) / ((x - center) ** 2 + (width / 2) ** 2)

    def Dopp_freq(self):
        """Analytical Doppler frequency (valid for slab geometry)."""
        dx = self.input.dx
        dt = 0.5 * dx / C
        dny = getattr(self.input, 'dny', 5)  # fallback to 5 if not in input
        k0 = 2 * PI * self.input.f0 / C
        kf = 2 * k0 * np.sin(np.abs(self.input.angle) * PI / 180)
        return dny * dx * kf / (2 * PI)

    def fit(self, func_type, fD, Freq, Power):
        """Perform Gaussian or Lorentzian fit."""
        x = Freq / fD
        if func_type.lower().startswith('lor'):
            p0 = [
                np.trapz(Power, x),
                x[np.argmax(Power)],
                np.ptp(x) / 2,
            ]
            popt, _ = curve_fit(self.lorentzian, x, Power, p0=p0, maxfev=10000)
            return self.lorentzian(x, *popt)
        else:
            p0 = [
                Power.max(),
                x[np.argmax(Power)],
                np.std(x),
            ]
            popt, _ = curve_fit(self.gaussian, x, Power, p0=p0, maxfev=10000)
            return self.gaussian(x, *popt)

    @classmethod
    def get_doppler_spectra(cls, subdir, f0, angle, mode, simulation, **kwargs):
        """Compute Doppler spectra and perform spectral fits."""
        fit = 'welch' if kwargs.get('welch_spectra', False) else 'FFT'
        simname = kwargs.pop('simname', None) or definitions.get_simname(f0, angle, mode, simulation, **kwargs)
        file = definitions.get_analysed_path(simname, subdir=subdir, fit=fit, **kwargs)

        if os.path.exists(file) and kwargs.get('load_if_existing', False):
            print(f'Loaded existing file: {file}', flush=True)
            return Struct.from_mat(file)

        FWD = cls(subdir, f0, angle, mode, simulation, **kwargs)
        output = FWD.output
        amp, phase = output.doppler_data[:, 0], output.doppler_data[:, 1]
        signal = amp * np.exp(-1j * phase)

        rem_left, rem_right = kwargs.get('rem_left', 0), kwargs.get('rem_right', 0)

        # Compute frequency spectrum
        if kwargs.get('welch_spectra', False):
            freqs, Sk = welch(signal, return_onesided=False)
        else:
            Sk = np.fft.fftshift(np.abs(np.fft.fft(signal)))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(signal)))

        # Remove center frequencies if requested
        idx0 = np.argmin(np.abs(freqs))
        Freqs = np.delete(freqs, slice(idx0 - rem_left, idx0 + rem_right))
        Signal = np.delete(Sk, slice(idx0 - rem_left, idx0 + rem_right))

        # Optional: remove extra frequencies
        fD = FWD.Dopp_freq()
        rem_f_extra = kwargs.get('rem_f_extra')
        if rem_f_extra is not None:
            rem_l_extra = np.atleast_1d(kwargs.get('rem_l_extra', 0))
            rem_r_extra = np.atleast_1d(kwargs.get('rem_r_extra', 0))
            rem_f_extra = np.atleast_1d(rem_f_extra)

            for fextra, l_ex, r_ex in zip(rem_f_extra, rem_l_extra, rem_r_extra):
                idx = np.argmin(np.abs(freqs - fextra * fD))
                Freqs = np.delete(Freqs, slice(idx - l_ex, idx + r_ex))
                Signal = np.delete(Signal, slice(idx - l_ex, idx + r_ex))

        # Perform fits
        lor_fit = FWD.fit('Lorentzian', fD, Freqs, Signal)
        gau_fit = FWD.fit('Gaussian', fD, Freqs, Signal)

        # Build output structure
        out = Struct(
            angle         = angle,
            k_perp        = 4 * PI * FWD.f0 / C * np.sin(FWD.angle * PI / 180) / 100,  # cm^-1
            fD = fD,
            fD_Sim        = Freqs[np.argmax(Signal)] / fD,
            fD_Lor        = Freqs[np.argmax(lor_fit)] / fD,
            fD_Gau        = Freqs[np.argmax(gau_fit)] / fD,
            freqs         = freqs,
            Freqs         = Freqs,
            Raw_signal    = Sk,
            Signal        = Signal,
            lorentian_fit = lor_fit,
            gaussian_fit  = gau_fit,
            f_mean        = np.mean([out.fD_Lor, out.fD_Gau]),
            f_std         = np.std([out.fD_Sim, out.fD_Lor, out.fD_Gau]),
            f_err         = max(abs(out.f_mean - f) for f in [out.fD_Sim, out.fD_Lor, out.fD_Gau]),
            validated     = kwargs.get('validated', 0),
        )
        # Save if requested
        if kwargs.get('save_results', False):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            Struct.to_mat(out, file)
            print(f'File saved: {file}', flush=True)

        return out

    def help(self):
        """Print available class methods."""
        methods = [m for m in dir(self) if callable(getattr(self, m)) and not m.startswith("__")]
        print("Available functions:")
        for m in methods:
            print(f" - {m}")


#%%
if __name__ == '__main__':
    subdir = 'linear_profile'
    spect, f0, angles, mode = 1, 60e9, [5], 'O'
    simulation = 'spectrum_study'

    for angle in angles:
        spectra = FW2DReader.get_doppler_spectra(
            subdir, f0, angle, mode, simulation,
            spect=spect, welch_spectra=False,
            save_results=False, load_if_existing=True,
            machine='irene'
        )
