import numpy as np
from numpy import pi, exp
from scipy.optimize import curve_fit, minimize, Bounds


# Define a Gaussian function
def gaussian(x, a, x0, sigma, sigma_is_FWHM=True):
    if sigma_is_FWHM:
        sigma = sigma / (2 * np.sqrt(2 * np.log(2)))
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# Define a Lorentzian function
def lorentzian(x, a, x0, FWHM):
    return a / ((x-x0)**2 / (FWHM/2)**2 + 1)
    
def taylor_inverse_FT(t, A, f0, tau_L, k2D):
    """
    Inverse FT of the Taylor function.
    """
    x=np.abs(t)/tau_L
    
    return A * np.exp(1j * 2 * np.pi * f0 * t) *  np.exp( - k2D * tau_L * (x -1 +np.exp(-x) ) ) 

def taylor(f, A, f0, tau_L, k2D, nfft=2048, dt=.5, ret_full=False):
    """
    Returns the Taylor function `T` interpolated to the given frequency vector `f`.
    Note that frequency values `f` outside the Nyquist range [-1/ (2 dt), 1/(2 dt)] yield NaNs.
    If ret_full is True, the full range of definition (`f_full`, `T_full`) is returned in addition to T`, otherwise returns only `T`.
    """
    # tmax = (nfft//2) - 0.5 * (1-nfft%2)
    t = np.arange(-nfft, nfft) * dt
    # print(t.min(), t.max(), np.min(np.abs(t)))
    C = taylor_inverse_FT(t, A, f0, tau_L, k2D)
    T_full = np.fft.fftshift(np.fft.fft(C))
    T_full = np.abs(T_full) * dt
    f_full = np.fft.fftshift(np.fft.fftfreq(t.shape[-1], d=dt))
    from scipy.interpolate import interp1d
    T = interp1d(f_full, T_full, kind='cubic', bounds_error=False)(f)
    if ret_full:
        return f_full, T_full, T
    
    else:
        return T

def taylor_fit_wrapper(xdata, ydata, noise, dt, gaussian_fit_params, lorentzian_fit_params,
                        nfft=2048,
                        bounds=Bounds([0, -30, 0, 0], [np.inf, 30, np.inf, 100]),
                        method='Nelder-Mead',
                        log_weight=0.6,
                        verbose=False):
    
    
    """
    Wrapper for the Taylor fit which uses the results of the Gaussian and Lorentzian fits as initial guess and performs
    a chi2 minimization alogrithm.
    
    Args:
    ----------
    xdata: array-like
        xdata to fit (normalized)
    ydata: array-like
        ydata to fit (normalized)
    noise: float
        noise level of the data (normalized)
    dt: float 
        time step of the FT (normalized according to xdata)
    gaussian_fit_params: array-like
        parameters of the Gaussian fit (A, x0, FWHM)
    lorentzian_fit_params: array-like
        parameters of the Lorentzian fit (A, x0, FWHM)
    bounds: array-like (optional)
        bounds for the fit parameters (A, x0, tau_L, k2D)
    method: str (optional)
        method to use for the minimization algorithm, see scipy.optimize.minimize for details
    log_weight: float (optional)
        weight of the logarithmic vs. the linear chi2 residuals in the objective function (between 0 and 1)
    verbose: bool (optional)
        whether to print the results of the minimization algorithm
        
    Returns:
    ----------
    result: scipy.optimize.OptimizeResult
        result of the minimization algorithm, see scipy.optimize.minimize for details
    model_function: function
        function which returns the model given the fit parameters
    
    """

    # Define your model function
    def _model_function(x, *params):
        return taylor(x, *params, nfft=nfft, dt=dt, ret_full=False)

    # Define the objective function to minimize (sum of squared residuals)
    def _objective_function(params, log_weight=0.5):
        
        def _chi2(data, model, noise, scale):
            if not scale in ['linear', 'log']:
                raise ValueError('scale must be either "linear" or "log"')
            
            if scale == 'linear':
                chi2 = np.sum( (model + noise - data)**2)
                # weigh by some estimate of the scatter in the data:
                chi2 /= np.sum(np.diff(data)**2)
            elif scale == 'log':
                chi2 = np.sum((np.log10(model + noise) - np.log10(data))**2)
                # weigh by some estimate of the scatter in the data:
                chi2 /= np.sum(np.diff(np.log10(data))**2)
            return chi2
            
        model = _model_function(xdata, *params)
        
        chi2log = _chi2(ydata, model, noise, scale='log')
        chi2lin = _chi2(ydata, model, noise, scale='linear')
        
        residuals =  log_weight * chi2log + (1-log_weight) * chi2lin
        
        return residuals
    
    # algorithm can be faster if good initial guess is provided, and sometimes a necessity to even converge:
    def _get_taylor_init_guess(x0_gaussian, FWHM_gaussian, x0_lorentzian, FWHM_lorentzian):
        
        # amplitude is of order unity:
        A = 1
        
        # center is trivial (take the mean)
        x0 = (x0_gaussian + x0_lorentzian) / 2
        
        # width is more complicated:
        k2D = 1.4 * 2 * np.pi * (FWHM_lorentzian / 2)
        k2U2 = (1 * 2 * np.pi * (FWHM_gaussian / 2)) ** 2 / 2 / np.log(2)
        
        tau_L = k2D / k2U2
        
        p0 = [A, x0, tau_L, k2D]
        
        return p0

    init_guess = _get_taylor_init_guess(*gaussian_fit_params[1:], *lorentzian_fit_params[1:])

    import time
    t0 = time.time()
    
    # Call minimize with given method
    result = minimize(
        _objective_function, 
        init_guess, 
        args=(log_weight), 
        bounds=bounds,
        method=method,# 'trust-constr', #None 
        tol=1e-5,
        options={'maxiter': 2000}
        )
    
    time_elapsed = time.time() - t0
    
    if verbose:
        print('Time elapsed: ', time_elapsed)
        print("Optimal parameters:", result.x)
        print("Function value:", result.fun)
        
    result.time_elapsed = time_elapsed
    
    return result, _model_function

    
def gauss_lorentz_fit_wrapper(xdata, ydata, curve_type, p0=None, verbose=False, **kwargs):
    
    # remove any NaNs from the data:
    mask = np.isnan(ydata) | np.isnan(xdata)
    xdata = xdata[~mask]
    ydata = ydata[~mask]
    

    if curve_type not in fitfuncs.keys():
        # Invalid curve type
        raise ValueError(f'Invalid curve type. Must be one of: {list(fitfuncs.keys())}')
    
    fitfunc = fitfuncs[curve_type]
    # initial guess for fit parameters
    bounds = kwargs.pop('bounds', None)
    
    if p0 is None:
        # Initial guess for the Gaussian fit parameters
        p0 = [np.max(ydata), np.mean(xdata), np.std(xdata)]
        
    if not bounds:
        # we just need to make sure that amplitude and FWHM are positive (otherwise, the subsequent Taylor fit might fail due to poor initial guess)
        bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])
    
    popt, _ = curve_fit(fitfunc, xdata, ydata, p0=p0, bounds=bounds, **kwargs)
    
    return fitfunc, popt


def perform_fits(xdata, ydata, noise, dt, p0=None, verbose=False, **kwargs):
    """
    Perform Gaussian, Lorentzian and Taylor fits (in this order) to the data.
    
    Args:
    ----------
    xdata: array-like
        xdata to fit (normalized)
    ydata: array-like
        ydata to fit (normalized)
    noise: float
        noise level of the data (normalized)
    dt: float 
        time step of the FT (normalized according to xdata)
        
    Returns:
    ----------
    fit_results: Struct
        dictionary containing the fit results
    """
    from matlabtools import Struct
    fit_results = Struct()
    

    # Gaussian and Lorentzian fits (easy)
    for curve_type in ['gaussian', 'lorentzian']:
        curve_func, curve_params = gauss_lorentz_fit_wrapper(xdata, ydata, curve_type, p0=p0)
        res = Struct({'func': curve_func, 'params': curve_params})
        fit_results.update({curve_type: res})

    # need the fit results to initialize the Taylor fit:
    frG = fit_results['gaussian']['params']
    frL = fit_results['lorentzian']['params']

    # taylor fit:
    _res, func = taylor_fit_wrapper(xdata, ydata, noise, dt, frG, frL, 
                                    log_weight=kwargs.get('log_weight', 0.6), 
                                    verbose=verbose)
    res = Struct({'func': func, 'params': _res.x,
                                'optimizeresult': _res})

    fit_results.update({'taylor': res})
    return fit_results


fitfuncs = {
    # 'odd': gaussian,
    'gaussian': gaussian,
    'lorentzian': lorentzian,
    'taylor': taylor,
}