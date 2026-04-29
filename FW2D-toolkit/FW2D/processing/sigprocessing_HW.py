#%%
from scipy.signal import welch
from fullwave2d.core.wrapper import InputData, OutputData
from FW2D import DataInterface
import numpy as np
from matlabtools import Struct
from pathlib import Path
import copy
import warnings
import traceback

##########################################################################
####   for HW dt = 0.1 * 1 / Omega_I                            ##########
####   for ST dt = 1e-4                                         ##########
##########################################################################



from scipy import constants as cnst
mi = 6 * cnst.m_p #kg
Te = 1800 #eV
B = 1.0 #T
cs = np.sqrt(cnst.e * Te / mi)
OmegaI = cnst.e * B / mi
rhos = cs / OmegaI
# from FW2D.processing.fit_utils import *
# from DBS.processing.encode import encode_mask, decode_mask
# %%
ODD_EVEN_THRESHOLD = 0.1

def get_normalized_complex_signal(x,y):
    """
    Returns the complex FW signal.
    """
    return x * np.exp(1j * y)

def PSD_welch(z, dt, nperseg=512, noverlap=256, axis=-1, ):
    """
    Returns the power spectral density of the signal Z, using the Welch method.
    """
    f, P = welch(z, fs=1/dt, return_onesided=False, nperseg=nperseg, noverlap=noverlap, axis=axis, scaling = 'density')
    f, P = np.fft.fftshift(f, axes=axis), np.fft.fftshift(P, axes=axis)
    return f, P

def PSD_raw(z, dt, axis=-1):
    """
    Returns the power spectral density of the signal Z, using the raw FFT.
    """
    n = z.shape[axis]
    f = np.fft.fftshift(np.fft.fftfreq(z.shape[axis], d=dt), axes=axis)
    P = np.fft.fftshift(np.abs(np.fft.fft(z, axis=axis)), axes=axis)**2 / (n / dt)
    return f, P


def get_noise_level(f, P, fmin=6.2e7, fmax=6.5e7, axis=-1):
    """
    Determine the noise level by averaging the PSD in a given frequency range. If P is multidimensional, the axis argument is assumed to correspond to the frequency.
    """
    cond = (f>fmin) & (f<fmax)
    P_cond = np.take(P, np.where(cond), axis=axis)
    P_noise = np.mean(P_cond, axis=axis)
    return P_noise[0]

# def get_noise_level(f, P, fmin=6.5e7, fmax=6.8e7, axis=-1):
#     cond = (f > fmin) & (f < fmax)
#     if not np.any(cond):
#         # fallback: use the top 5% of frequencies
#         n = len(f)
#         top5 = max(1, int(0.05 * n))
#         P_noise = np.mean(np.sort(P.ravel())[-top5:])
#         warnings.warn(
#             f"Noise frequency range [{fmin:.2e}, {fmax:.2e}] Hz not covered by FFT "
#             f"(f_max={np.max(np.abs(f)):.2e} Hz). "
#             f"Falling back to top-5% PSD estimate: P_noise={P_noise:.2e}"
#         )
#         return P_noise
#     P_cond = np.take(P, np.where(cond), axis=axis)
#     P_noise = np.mean(P_cond, axis=axis)
#     return P_noise[0]

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

def get_center_of_gravity(f, P)->np.ndarray:
    """
    Returns the center of gravity of the spectrum.
    """
    f_cog = nantrapz(f * P, x=f) / nantrapz(P, x=f)
    return f_cog

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
    
def get_spikes_in_PSD(f, P, threshold=3, axis=-1):
    """
    Returns a boolean mask indicating the frequencies identified as coinciding with a spurious spike of the PSD.
    The threshold is with respect to the standard deviation of diff(log10(P[dB])) and diff(diff(log10(P[dB]))).
    """
    
    SdB=10*np.log10(P) # take the PSD in dB 
    
    # define 1st and second order differences (keeping the same dimensions as f):
    dSdB=np.diff(SdB, append=SdB[-1])
    ddSdB=np.diff(dSdB, prepend=dSdB[0])
    
    # use a combination of three conditions to identify spikes
    cond1_plus  = dSdB >  threshold * np.std(dSdB)
    cond1_minus = dSdB < -threshold * np.std(dSdB)
    cond2       = ddSdB < -threshold * np.std(ddSdB)
    
    is_spike_mask   = cond1_plus | cond1_minus | cond2
   
    return is_spike_mask

def get_frequency_range(f, fmin, fmax):
    return (np.abs(f)>fmin) & (np.abs(f)>fmax)

# ST [0.5*4.9e3, 4.9e3]
# HW [6.5e7, 6.8e7]
def preprocessing(z, dt, noise_freq_range = [6.5e7, 6.8e7], nperseg=512, noverlap=256, fmin=100, fmax=50, spikes_removal_threshold=3, **kwargs):
    """Compute PSD and related spectral quantites of a given complex signal z. Returns a `specobj` structure containing the results."""
    
    f_raw, P_raw = PSD_raw(z, dt)
    f, P         = PSD_welch(z, dt, nperseg=nperseg, noverlap=noverlap)
    P_noise      = get_noise_level(f_raw, P_raw, *noise_freq_range).squeeze()

    # remove frequencies below/above some threshold:
    include_mask = get_frequency_range(f, fmin, fmax) # convert to int to avoid problems when exporting/importing 

    # remove spikes:
    is_spike_mask = get_spikes_in_PSD(f, P, threshold=spikes_removal_threshold)
    include_mask = include_mask & ~is_spike_mask

    # symmetric/asymmetric components of the spectrum
    P_even, P_odd, fsym = get_even_odd_spectrum(f,P, assume_centered=False)
    P_odd_positive      = ( P_odd + np.abs(P_odd) ) / 2

    # center of gravity of the asymmetric spectrum:
    f_cog = get_center_of_gravity(fsym, P_odd_positive)
    # f_cog_PH = get_center_of_gravity(diag_data.fsym, diag_data.P_dop)

    # compute the power of the odd and even components:
    power_odd  = nantrapz(P_odd_positive, x=fsym)
    power_even = nantrapz(P_even, x=fsym)
    NormS      = nantrapz(P, x = f)
    rms        = z.std() ** 2

    specobj = Struct()
    specobj.validated      = 0
    specobj.dt             = dt
    specobj.f              = f
    specobj.P              = P
    specobj.include_mask   = include_mask
    specobj.f_rawFFT       = f_raw
    specobj.P_rawFFT       = P_raw
    specobj.P_noise        = P_noise
    specobj.fsym           = fsym
    specobj.P_even         = P_even
    specobj.P_odd          = P_odd
    specobj.P_odd_positive = P_odd_positive
    specobj.f_cog          = f_cog
    specobj.power_odd      = power_odd
    specobj.power_even     = power_even
    specobj.NormS          = NormS
    specobj.rms            = rms
    specobj.odd_even_threshold = ODD_EVEN_THRESHOLD
    
    return specobj

def get_data_interface_for_specobj(specobj):
    h = specobj.header
    data_interface = DataInterface(h.subdir)
    return data_interface

def init_specobjs(data_interface, isims, signal_include_mask=None, verbose=False, nperseg=512, noverlap=256, **kwargs):
        
    # shorthands:
    par = data_interface.params    
    specobjs = []
    
    for isim in isims:

        # add some metadata:
        header = Struct()
        header.machine = data_interface.machine
        header.subdir  = data_interface.subdir 
        header.freqGHz = par.F[isim-1]
        header.isim    = isim
        header.mode    = par.mode        

        t, x, y = data_interface.get_signal(isim)
        dt = 0.1 * (1 / OmegaI)  # HW
        z = get_normalized_complex_signal(x, y)
        
        noise_freq_range = [6.5e7, 6.8e7] # HW
        
        
        specobj = Struct()
        specobj.header = header
        specobj.update(preprocessing(z, dt, noise_freq_range,                                
                                     nperseg=z.size, noverlap=z.size // 2, **kwargs))
                
        specobjs.append(specobj)
        
    return specobjs

def perform_specobj_fits(specobj, include_mask=None, p0=None, reinitialize=False, signal_include_mask=None, verbose=False):
    
    from FW2D.processing import perform_fits
    s = specobj  # shortcut
    
    if include_mask is None:
        include_mask = s.include_mask  #get_frequency_range(s.f, fmin=50, fmax=50)
    
    include_mask = include_mask.astype(bool)
    
    if not hasattr(s, 'fsym'):
        reinitialize = True
    
    if reinitialize:
        dataI = get_data_interface_for_specobj(specobj)
        updated_specobj = init_specobjs(dataI, isims=[specobj.header.isim], signal_include_mask=signal_include_mask)[0]
        specobj.update(updated_specobj)
        perform_specobj_fits(specobj, include_mask=include_mask, p0=p0, reinitialize=False, verbose=verbose)
        
    xscale = 1e7 # HW
    yscale = np.nanmax(s.P)
    xdata  = s.f[include_mask] / xscale
    ydata  = s.P[include_mask] / yscale
    noise  = s.P_noise / yscale
    s.include_mask = include_mask

 
    s.xscale = xscale
    s.yscale = yscale

    try:

        fit_results_full = Struct()

        xdata_odd        = s.fsym / xscale
        ydata_odd        = s.P_odd_positive / yscale
        from FW2D.processing.fit_utils import gauss_lorentz_fit_wrapper
        curve_func, curve_params = gauss_lorentz_fit_wrapper(xdata_odd, ydata_odd,curve_type='gaussian')
        # res_odd = Struct({'func': curve_func, 'params': curve_params})
        # fit_results_full.update({'odd': res_odd})
            
        if p0 is None:
            cog = get_center_of_gravity(xdata, ydata)
            if np.abs(np.abs((xdata[np.argmax(ydata)] / xscale - cog )) / (xdata[np.argmax(ydata)] / xscale)) > 0.2:
                cog = xdata[np.argmax(ydata)] 

            p0 = [np.nanmax(ydata), cog, 0.5]
       
        try: 
            fit_results_full.update(perform_fits(xdata, ydata, noise, s.dt * xscale, p0=p0))
        
            if verbose:
                print(f'results Gauss:\n', fit_results_full.gaussian.params)
                print('results Lorentz\n', fit_results_full.lorentzian.params)
            s.fit_params = Struct()
            for curve_type, fit_result in fit_results_full.items():
                s.fit_params[curve_type] = fit_result['params']
            
        except RuntimeError:
            if verbose:
                print(f'Fitting failed for specobj with header:\n{s.header}')
    
    except Exception as e:
        traceback.print_exc()
        warnings.warn(f'Fitting failed for specobj with header:\n{s.header}')  
            
def get_fDop_from_fit_results(fit_params, xscale=1e7, f_cog=None):
    """Returns a dictionary containing the estimated Doppler frequency and errors in Hz, using xscale as the conversion factor between the normalized fit range and the frequency.
    Args:
    -----
        fit_params: dict
            Dictionary containing the fit params of the different curves.
        xscale: float
            Scaling factor for the frequency.
    """
    fDop_fits = np.array([params[1] for params in fit_params.values()]) * xscale
    
    if f_cog is not None:
        fDop_fits = np.append(fDop_fits, f_cog)
    
    try:
        if len(fDop_fits) <= 1 or np.isnan(fDop_fits).sum() >= len(fDop_fits)-1: # avoid underestimating the spread if we only have one valid estimation
            fDop_spread = np.nan
        else:
            fDop_spread = np.nanmax(fDop_fits) - np.nanmin(fDop_fits)
            
        fDop_median = np.nanmedian(fDop_fits)
    except ValueError:
        fDop_spread = np.nan
        fDop_median = np.nan
        
    if fDop_median > 0:
        fDop_max = np.nanmax(fDop_fits)
        fDop_min = np.nanmin([np.nanmin(fDop_fits), fDop_median - 0.5 * (fDop_max - fDop_median)])
    else:
        fDop_min = np.nanmin(fDop_fits)
        fDop_max = np.nanmax([np.nanmax(fDop_fits), fDop_median + 0.5 * (fDop_median - fDop_min)])
        
    
    outp = {
        'fDop': fDop_median,
        'fDop_max': fDop_max ,
        'fDop_min': fDop_min ,
        'dfDop': fDop_max - fDop_min,
    }
    for curve_type, params in fit_params.items():
        outp[f'fDop_{curve_type}'] = params[1] * xscale
    
    
    return outp 

def make_title(ax, data_interface, isim):
    par = data_interface.params
    freq, theta, waist, mode = par.F[isim-1], par.theta[isim-1], par.waist[isim-1], par.mode[isim-1]

    title = r'f = %.1f GHz, theta = %d°, waist = %d, %s mode' %(freq, theta, waist, mode)
    ax.set_title(title)
    
def plot_semi_log10(ax,x,y, *args, **kwargs):
    return ax.plot(x,10*np.log10(y), *args, **kwargs)

def show_spec(specobj, ax=None, axis_labels=True, 
              legend=True,
              show_curves=['rawPSD', 'fittedPSD', 'oddPSD', 'fit_odd', 'fit_gaussian', 'fit_lorentzian', 'fit_taylor', 'annotations'],
              verbose=False,
              indicate_estimate=False,
              **plot_kwargs):
    """ Plots the spectral data contained in specobj.
    :param specobj: object containing the spectral data
    :param ax: matplotlib (or other) axis object to plot the data
    :param axis_labels: whether to show axis labels
    :param legend: whether to show the legend
    :param show: list of strings indicating which curves to show
    :return: plot_dict: dictionary containing the plot objects
    """
    colors = {
            'gaussian': 'blue',
            'lorentzian': 'darkgreen',
            'taylor': 'black',
            'odd': 'magenta',
            'even': 'lightgreen'
        }
    import matplotlib.pyplot as plt
    from FW2D.processing.fit_utils import fitfuncs
    plot_dict = {}    
    s = specobj # shortcut
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (5, 4))
    
    include_mask = s.include_mask == 1
    xscale = s.xscale
    yscale = s.yscale
    xdata = s.f[include_mask] / xscale
    ydata = s.P[include_mask] / yscale
    
    for show_c in show_curves:
        
        if show_c == 'rawPSD':
            # determine the bounds of the plot:
            fmin, fmax = np.min(xdata), np.max(xdata)
            ind_f = (s.f > fmin) & (s.f  < fmax)
            _xraw, _yraw = np.copy(s.f), np.copy(s.P)
            # _xraw[~ind_f] = np.nan
            # _yraw[~ind_f] = np.nan
            
            lab = plot_kwargs.pop('label', 'raw PSD')
            lrawPSD = plot_semi_log10(ax, _xraw / xscale * 10, _yraw + s.P_noise, label=lab, alpha=0.7, **plot_kwargs)
            plot_dict['lrawPSD'] = lrawPSD
            
        elif show_c == 'fittedPSD':
            lab = plot_kwargs.pop('label', 'fitted PSD')
            lfittedPSD = plot_semi_log10(ax, xdata * 10 , ydata * yscale + s.P_noise, color='red', label=lab, **plot_kwargs)
            plot_dict['lfittedPSD'] = lfittedPSD
        
        # elif show_c == 'oddPSD' and hasattr(s, 'fsym'):
        #     lab = plot_kwargs.pop('label', 'odd PSD')
        #     loddPSD = plot_semi_log10(ax, s.fsym / xscale, s.P_odd_positive + s.P_noise, color=colors['odd'], label=lab, ls='--', **plot_kwargs)
        #     plot_dict['loddPSD'] = loddPSD
        
        # elif show_c == 'evenPSD' and hasattr(s, 'fsym'):
        #     lab = plot_kwargs.pop('label', 'even PSD')
        #     levenPSD = plot_semi_log10(ax, s.fsym, s.P_even, color=colors['even'], label=lab, ls='--', **plot_kwargs)
        #     plot_dict['levenPSD'] = levenPSD

        elif show_c in ['fit_odd', 'fit_gaussian', 'fit_lorentzian', 'fit_taylor']:
            curve_type = show_c.split('_')[1]
            if f'lfit_{curve_type}' in plot_dict.keys(): # skip if already plotted
                if verbose:
                    print(f'skipping {show_c} as it is already plotted')
            else:
                if hasattr(s, 'fit_params') and curve_type in s.fit_params.keys():                    
                    
                    curve_func   = fitfuncs[curve_type]
                    curve_params = s.fit_params[curve_type]
                    xfit = np.linspace(np.min(xdata), np.max(xdata), 1000)
                    
                    
                    if curve_type == 'taylor':
                        yfit = curve_func(xfit, *curve_params, dt=s.dt * xscale)
                    else:
                        yfit = curve_func(xfit, *curve_params)
                        
                    x = xfit *10
                    y = yfit * yscale + s.P_noise
                    l = plot_semi_log10(ax, x, y, label=f'{curve_type} fit',color=colors[curve_type], **plot_kwargs)
                    plot_dict[f'lfit_{curve_type}'] = l
                
                else:
                    if verbose:
                        print(f'No fit results found for {curve_type}, skipping')
            
            
    

    if not hasattr(s, 'fit_params'):
        if verbose:
            print('No fit results found, skipping fit plots')
    else:
        
        if 'annotations' in show_curves:
    
            fDop_fits = np.array([fitpar[1] for fitpar in s.fit_params.values()]) * xscale
            
            # add line indicating the Center of Gravity of the odd part of the spectrum:
            cog_vline = ax.axvline(s.f_cog/xscale * 10, color=colors['odd'], ls='-.', label='odd CoG')
            
            # indicate noise level:
            noise_hline = ax.axhline(10*np.log10(s.P_noise), color='black', ls='--', alpha=0.5, lw=0.5, label='noise level')
            
            plot_dict['cog_vline'] = cog_vline
            plot_dict['noise_hline'] = noise_hline
            
            if indicate_estimate:
                plot_dict.update(show_estimate_with_errorbars(ax, s, **plot_kwargs))
        
    
    ax.set_ylim(bottom=10*np.log10(0.5*s.P_noise))
    
    if axis_labels:
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('PSD [dB]')
    
    if legend:
        ax.legend()
    
    return plot_dict


def show_estimate_with_errorbars(ax, specobj, **kwargs):
    
    col = kwargs.pop('color', 'darkblue')
    lw = kwargs.pop('lw', 1.0)
    
    if not hasattr(specobj, 'fDop'):
        results = get_fDop_from_fit_results(specobj.fit_params, specobj.xscale)
        fDop = results['fDop']
        fDop_min = results['fDop_min'] 
        fDop_max = results['fDop_max'] 
    else:
        fDop, fDop_min, fDop_max = specobj.fDop, specobj.fDop_min, specobj.fDop_max
    
    # indicate the horizontal error bar somewhere over the top of the Doppler peak:
    Ampl_lorentzian = specobj.fit_params['lorentzian'][0]
    y_errorbar = Ampl_lorentzian * 1.2 * specobj.yscale
    
    if np.isnan(fDop):
        return {'fDop_vline': None, 'dfDop_hline': None}
        
        
    # add line indicating the median of the fits:
    fDop_vline = ax.axvline(fDop/1e7, color=col, lw=lw, ls='--', **kwargs)
    dfDop_hline = plot_semi_log10(ax,np.array([fDop_min, fDop_max]) / 1e7, y_errorbar * np.array([1,1]), color=col, lw=lw, **kwargs)
    return {'fDop_vline': fDop_vline, 'dfDop_hline': dfDop_hline}

class OutputWrapper():
    
    def __init__(self, data_interface, use_existing_file=True):
        
        self.data_interface = data_interface
        
        
        # check if file already exists, in which case we load that data:
        self.outpath = self._get_outpath()
        self.merged_specobj = self._check_existing_file(self.outpath) if use_existing_file else None        
        
        # if not, we create a new placeholder object:
        if self.merged_specobj is None:
            self.merged_specobj = self._merged_placeholder()
            
        # update the most relevant data at the top-level of the file
        self._make_toplevel_data(self.merged_specobj)
        
    def __getattr__(self, attr):
        """Shortcut to access the attributes of the merged_specobj object instead of the OutputWrapper object."""
        try:
            return self.merged_specobj.__getattr__(attr)
        except AttributeError:
            return super().__getattr__(attr)

    def __repr__(self):
        txt = """Wrapper for exporting spectral data.\n"""
        
        if hasattr(self, 'merged_specobj'):
            txt += f'Contains object merged_specobj wit hattributes:\n-------------------\n'
            txt += self.merged_specobj.__repr__()
        return txt
    
    # def __getattr__(self, __name: str) -> Any:
    #     return self.merged_specobj.__getattr__(__name)
    
    def _make_header(self, specobj) -> Struct:
        
        _header = Struct()
        _header.description = 'Collection of spectral DBS data and fits.'
        _header.update(Struct(specobj.header.copy()))
        for attr in ['time', 'freqGHz', 'isim']:
            _header.pop(attr, None)
        return _header
    
    def make_header(data_interface):
    
        # shorthands:
        par    = data_interface.params
        isims = np.arange(1, par.Nbsim + 1)

        # add metadata:
        header = Struct()
        header.machine = data_interface.machine
        header.freqGHz = par.F[isims-1]        
        header.Nbsim   = par.Nbsim
        
        return header
    
    def update_specobjs(self, specobjs):
        specobjs = self._handle_attributes(specobjs)

        # convert to list if it's a numpy array (happens when loaded from .mat file)
        if not isinstance(self.merged_specobj.specobjs, list):
            self.merged_specobj.specobjs = list(self.merged_specobj.specobjs)

        # expand the list if new simulations were added beyond the original Nbsim
        max_isim = max(s.header.isim for s in specobjs if hasattr(s, 'header'))
        while len(self.merged_specobj.specobjs) < max_isim:
            self.merged_specobj.specobjs.append({})

        for specobj in specobjs:
            if specobj is not None:
                self.merged_specobj.specobjs[specobj.header.isim - 1] = specobj

        self._make_toplevel_data(self.merged_specobj)
    
    def export(self, outpath=None, verbose=False, use_existing_file=True):
        """Exports the spectral data contained in specobj to a file.
        """
        if outpath is None:
            self.outpath = self._get_outpath()
        
        outdir = self.outpath.parent
        if not outdir.exists():
            outdir.mkdir(parents=True)
            # print(f"Created directory: {outdir}")
        
        # export to matlab (using the simple hack to put everything in a single attribute of a dict)
        _ = Struct()
        _.outp = self.merged_specobj
        _.to_mat(self.outpath)
        

        if verbose:
            print (f'Exported to {self.outpath}')
         
    def _merged_placeholder(self) -> Struct:
        """Creates a placeholder object for the merged spectral data.
        """
        merged_specobj = Struct()           
        merged_specobj.header = OutputWrapper.make_header(self.data_interface)
        merged_specobj.specobjs = [{}] * merged_specobj.header.Nbsim
        return merged_specobj
                
    def _get_outpath(self)->Path:
        from fullwave2d import definitions as defs
        from pathlib import Path
        dataI = self.data_interface
        DATA_FDOP_DIR = Path('/home/FO278650/Bureau/FullWave2D_FO/data/fw2d/fDop_estimation')
        outpath = DATA_FDOP_DIR / f'{dataI.subdir}.mat' #defs.DATA_FDOP_DIR
        return outpath
        
    def _handle_attributes(self, specobjs)->None:
        
        exclude_attrs = ['f_rawFFT', 'P_rawFFT',
                         'header.params', 'fit_results_full',
                         'signal_include_mask']
        
        # make a copy to avoid modifying the original objects
        _specobjs = copy.deepcopy(specobjs)
        
        for specobj in _specobjs:
            
            for attr in exclude_attrs:
                if hasattr(specobj, attr):
                    
                    delattr(specobj, attr)                   
                    
        return _specobjs

    def _check_existing_file(self, outpath, verbose=True)-> Struct:
        if Path(outpath).exists():
            if verbose:
                print(f'Found existing file {outpath}, loading data from file.')
            
            merged_specobj = Struct.from_mat(
                outpath, mat_key='outp')
            
            for s in merged_specobj.specobjs: # this is necessary because it is written as int, but indexing with ones and zeros is not the same as indexing with True and False
                
                if hasattr(s, 'include_mask'):
                    s.include_mask = s.include_mask.astype(bool)

            return merged_specobj
        else:
            return None
    
    
    def _make_toplevel_data(self, merged_specobj):

        par     = self.data_interface.params
        isims   = np.arange(1, par.Nbsim + 1)
        freqGHz = par.F[isims - 1]

        # use the actual number of specobjs slots, not the stale header.Nbsim
        Nbsim = len(merged_specobj.specobjs)

        validated = np.zeros(Nbsim)
        fDop      = np.nan * np.ones(Nbsim)
        dfDop     = np.copy(fDop)
        fDop_max  = np.copy(fDop)
        fDop_min  = np.copy(fDop)
        theta     = par.theta
        waist     = par.waist

        # also update the header to reflect the new count
        merged_specobj.header.Nbsim = Nbsim

        indices_treated = []

        from FW2D.io.utils import is_iterable
        if not is_iterable(merged_specobj.specobjs):
            merged_specobj.specobjs = [merged_specobj.specobjs]

        for i, specobj in enumerate(merged_specobj.specobjs):

            if specobj is None:
                continue

            if len(specobj) != 0:
                validated[i] = specobj.validated

                if validated[i] != 0.0:
                    indices_treated.append(i + 1)

                if hasattr(specobj, 'fDop'):
                    fDop[i] = specobj.fDop
                if hasattr(specobj, 'fDop_max') and hasattr(specobj, 'fDop_min'):
                    fDop_max[i] = specobj.fDop_max
                    fDop_min[i] = specobj.fDop_min
                    dfDop[i]    = np.abs(fDop_max[i] - fDop_min[i])
                else:
                    if hasattr(specobj, 'dfDop'):
                        dfDop[i] = specobj.dfDop

        # update freqGHz/theta/waist to match the new Nbsim if par was also extended
        merged_specobj.freqGHz   = par.F[:Nbsim]
        merged_specobj.theta     = par.theta[:Nbsim] if len(par.theta) >= Nbsim else par.theta
        merged_specobj.waist     = par.waist[:Nbsim] if len(par.waist) >= Nbsim else par.waist
        merged_specobj.fDop      = fDop
        merged_specobj.dfDop     = dfDop
        merged_specobj.fDop_min  = fDop_min
        merged_specobj.fDop_max  = fDop_max
        merged_specobj.validated = validated

        merged_specobj.header.isims_treated = np.array(indices_treated)
# %%
if __name__ == '__main__':
    
    subdir = 'HW_kscan_well_C1.0'
    dataI  = DataInterface(subdir)
    wrapper = OutputWrapper(dataI)

    # isims = dataI._get_sim_choice([7])
    # specobjs = init_specobjs(dataI, isims=isims)
    
    # perform_specobj_fits(specobjs[0])
    

# %%

    
    t, x, y = dataI.get_signal(3)
    dt   = 0.1 * 1 / (OmegaI)  #1e-4
    z = get_normalized_complex_signal(x, y)
    noise_freq_range = [6.5e7, 6.8e7]
    specobj = Struct()
    specobj.update(preprocessing(z, dt, noise_freq_range,                                
                                 nperseg=z.size, noverlap=z.size // 2, ))
    perform_specobj_fits(specobj)
    

    show_spec(specobj, )
    # fd = get_fDop_from_fit_results(specobj.fit_params)['fDop']
# %%
    from matlabtools import Struct
    fdop = Struct.from_mat('/home/FO278650/Bureau/FullWave2D_FO/data/fw2d/fDop_estimation/HW_rad_scan_C1.0.mat', 'outp')

# %%
    f_raw, P_raw = PSD_raw(z, dt)
    f, P         = PSD_welch(z, dt, nperseg=512, noverlap=256)
    P_noise      = get_noise_level(f_raw, P_raw, *[6.5e7, 6.8e7]).squeeze()
    
    import matplotlib.pyplot as plt 
    plt.plot(f_raw, 10 * np.log10(P_raw), c = 'b')
    plt.plot(f_raw, 10 * np.log10(P_raw + P_noise), c = 'k')
    plt.plot(f, 10 * np.log10(P + P_noise), c = 'r')

# %%
    kp = 4 * np.pi * 46.0 * 1e9 / cnst.c * np.sin(10 * np.pi / 180) / 100 # cm-1

    print(2 * np.pi * fd / kp)
# %%
