#%%
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import warnings
import traceback
from scipy.signal import welch
from fullwave2d.core.wrapper import InputData, OutputData
from matlabtools import Struct
from fullwave2d.config.definitions import FW2D_irene_DATA_DIR, FW2D_marconi_DATA_DIR, FW2D_DATA_DIR
from data_source.fit_utils import *



def plot_semilog10(xdata, ydata, ax = None, **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize = (5, 4))
    ax.plot(xdata, 10 * np.log10(ydata), **plot_kwargs)
    
def nantrapz(y, x=None, ret_mask=False, **kwargs):
    """Wrapper for np.trapz that ignores NaNs contained in y or x"""
    # remove any NaNs:
    if x is not None:
        mask = np.isnan(y) | np.isnan(x)
        _y = y[~mask]
        _x = x[~mask]
    else:
        mask = np.isnan(y)
        _y = y[~mask]
    
    if not ret_mask:
        return(np.trapz(_y, x=_x, **kwargs))
    else:
        return(np.trapz(_y, x=_x, **kwargs), mask)
    
def PSD_welch(z, dt, nperseg = 512, noverlap = 256, axis = -1):
    
    f, P = welch(z, fs = 1 / dt, return_onesided=False, nperseg=nperseg, noverlap=noverlap, axis = axis, scaling = 'density')
    f, P = np.fft.fftshift(f, axes=axis), np.fft.fftshift(P, axes=axis)
    return f, P

def PSD_raw(Z, dt, axis=-1):
    """
    Returns the power spectral density of the signal Z, using the raw FFT.
    """
    n = Z.shape[axis]
    f = np.fft.fftshift(np.fft.fftfreq(Z.shape[axis], d=dt), axes=axis)
    P = np.fft.fftshift(np.abs(np.fft.fft(Z, axis=axis)), axes=axis)**2 / (n / dt)

    return f, P

def get_noise_level(f, P, fmin=0.5*4.9e3, fmax=4.9e3, axis=-1):
    """
    Determine the noise level by averaging the PSD in a given frequency range. If P is multidimensional, the axis argument is assumed to correspond to the frequency.
    """
    cond = (f>fmin) & (f<fmax)
    P_cond = np.take(P, np.where(cond), axis=axis)
    P_noise = np.mean(P_cond, axis=axis)
    return P_noise[0]

def get_noncentered_even_odd_decomposition(Y, f):
    """
    Decompose a function into its even and odd parts 
    without assuming that the x-axis is centered at 0.
    Works horizontally by default.
    """

    fnm = min(f)
    knm = 0
    knM = np.where(f < 0)[0][-1]
    fnM = f[knM]

    kpm = np.where(f >= 0)[0][0]
    fpm = f[kpm]
    fpM = max(f)
    kpM = len(f) - 1

    if fpM > -fnm:
        minkn = np.where(f == -fnm)[0][0]
        maxkp = knm
    else:
        minkn = kpM
        maxkp = np.where(f == -fpM)[0][0]

    if fpm < -fnM:
        minkp = knM
        maxkn = np.where(f == -fnM)[0][0]
    else:
        minkp = np.where(f == -fpm)[0][0]
        maxkn = kpm

    fp = -f[np.arange(minkp, maxkp-1, step=-1)]
    fn = -f[np.arange(minkn, maxkn-1, step=-1)]
    fsym = np.concatenate((fn, fp))

    Ytn = Y[np.arange(minkn, maxkn-1, step=-1)]
    Ytp = Y[np.arange(minkp, maxkp-1, step=-1)]
    Yt = np.concatenate((Ytn, Ytp))

    YYn = Y[maxkp:minkp+1]
    YYp = Y[maxkn:minkn+1]
    YY = np.concatenate((YYn, YYp))
    fsym2 = np.concatenate((f[maxkp:minkp+1], f[maxkn:minkn+1]))

    Y_even = 0.5 * (YY + Yt)
    Y_odd = 0.5 * (YY - Yt)

    return Y_even, Y_odd, fsym

def get_even_odd_spectrum(f, P, axis=-1, assume_centered=True):
    """
    Returns the symmetric and asymmetric part of the spectrum. If P is multidimensional, the axis argument is assumed to correspond to the frequency.
    """
    
    if assume_centered:
        P_even = (P + np.flip(P, axis=axis)) / 2
        P_odd = (P - np.flip(P, axis=axis)) / 2
        return P_odd, P_even, f # here, f is nott used but we return it for consistency with the other function
    else:
        P_even, P_odd, fsym = get_noncentered_even_odd_decomposition(P, f)
        return P_even, P_odd, fsym

def get_center_of_gravity(f, P)->np.ndarray:
    """
    Returns the center of gravity of the spectrum.
    """
    f_cog = nantrapz(f * P, x=f) / nantrapz(P, x=f)
    return f_cog
    
def get_subfolders(folder_path):
    return [f for f in folder_path.iterdir() if f.is_dir()]

def preprocessing(z, dt, nperseg = 512, noverlap = 256,):
    rms          = z.std() ** 2
    f_raw, P_raw = PSD_raw(z, dt)
    f, P         = PSD_welch(z, dt, nperseg = nperseg, noverlap=noverlap)
    P_noise             = get_noise_level(f_raw, P_raw)
    P_even, P_odd, fsym = get_even_odd_spectrum(f,P, assume_centered=False)
    P_odd_positive      = ( P_odd + np.abs(P_odd) ) / 2
    f_cog               = get_center_of_gravity(fsym, P_odd_positive)
    power_odd           = nantrapz(P_odd_positive, x=fsym)
    power_even          = nantrapz(P_even, x=fsym)
    NormS               = nantrapz(P, x = f)
    
    specobj = Struct()
    specobj.dt = dt
    specobj.f  = f
    specobj.P  = P
    specobj.f_rawFFT = f_raw
    specobj.P_rawFFT = P_raw
    specobj.P_noise = P_noise

    # specobj.include_mask = include_mask

    specobj.fsym = fsym
    specobj.P_even = P_even
    specobj.P_odd = P_odd
    specobj.P_odd_positive = P_odd_positive
    specobj.rms   = rms
    specobj.NormS = NormS
    specobj.power_odd = power_odd
    specobj.power_even = power_even
    specobj.f_cog = f_cog
    specobj.odd_even_threshold = 0.1
    
    return specobj

def init_specobjs(subdir, **kwargs):
    
    machine = kwargs.pop('machine', None)
    DATA_DIR = Path(FW2D_irene_DATA_DIR) if machine == 'irene' else Path(FW2D_marconi_DATA_DIR) if machine == 'marconi' else Path(FW2D_DATA_DIR)    
    
    folders = get_subfolders(DATA_DIR.joinpath(subdir))
    specobjs = []


    for simname in folders:
        
        inp = InputData.load_pickle(simname, subdir=subdir, machine = 'irene')
        out = OutputData(simname, subdir=subdir, machine = 'irene')
        
        header         = Struct()
        header.simname = simname
        header.f0      = int(inp.f0 * 1e-9)
        header.angle   = -inp.angle
        header.mode    = inp.mode
        header.waist   = int(inp.waist / inp.dx -  int((inp.TFSF / 2 - 4))) # in units of dx
        header.name    = inp.name
        header.nperseg = out.doppler_data.shape[0]  

        specobj = Struct()
        specobj.header = header
        amp, phase = out.doppler_data[:, 0], out.doppler_data[:, 1]
        z          = amp * np.exp(1j * phase)
        dt         = 1e-4
        specobj.update(preprocessing(z, dt = dt, nperseg=z.size, noverlap=z.size // 2))
        specobjs.append(specobj)
    
    return specobjs
    
def perform_specobj_fits(specobj, p0 = None):
    
    s = specobj
    xscale = 1e3
    yscale = np.nanmax(s.P)
    xdata  = s.f / xscale
    ydata  = s.P / yscale
    noise  = s.P_noise / yscale

 
    s.xscale = xscale
    s.yscale = yscale

    try:
        fit_results_full = Struct()
        
        xdata_odd        = s.fsym / xscale
        ydata_odd        = s.P_odd_positive / yscale
        curve_func, curve_params = gauss_lorentz_fit_wrapper(xdata_odd, ydata_odd, curve_type='gaussian')
        # res_odd = Struct({'func': curve_func, 'params': curve_params})
        # fit_results_full.update({'odd': res_odd})
        
        cog = get_center_of_gravity(xdata, ydata)
        
        if np.abs(np.abs((s.f[np.argmax(s.P)] / xscale - cog )) / (s.f[np.argmax(s.P)] / xscale)) > 0.15:
            cog = s.f[np.argmax(s.P)] / xscale
        
        p0 = [np.nanmax(ydata), cog, 0.5e2 / xscale] # initial guess for the Gaussian fit parameters
        fit_results_full.update(perform_fits(xdata, ydata, noise, s.dt * xscale, p0=p0))
        
        s.fit_params = Struct()
        for curve_type, fit_result in fit_results_full.items():
            s.fit_params[curve_type] = fit_result['params']
            
    except Exception as e:
        traceback.print_exc()
        warnings.warn(f'Fitting failed for specobj with header:\n{s.header}')  
        