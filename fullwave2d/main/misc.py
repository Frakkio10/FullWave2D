import numpy as np
from numpy import tanh, sin, cos, pi
import scipy.constants as co

def find_nearest(array, value):
    """
    Find the the element in array that is closest to the value argument.
    Returns the index and value of that element.
    """
    if type(array) != np.ndarray:
        array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def tanh_transition(x, y1, y2, a, b):
    """ Hyperbolic tangent transition between two values y1, y2
    at a point a, with width b.
    Args:
        x (float or array-like):
            coordinates on which to evaluate this function.
        y1, y2 (float): values at -/+ infinity
        a (float): transition point, i.e. y''(a) = 0
        b (float): determines how smooth the transition is: high b -> sharp transition
    Returns:
        y (float or array-like):
            the function evaluated at point(s) x
    """
    y = y1 + (y2 - y1) * 0.5 * (1 + tanh((x - a) / b))
    return y

def _lin_transition(x, x1, x2, y1, y2):
    if x < x1:
        return y1
    elif x > x2:
        return y2
    else:
        return y1 + (y2 - y1) / (x2 - x1) * (x - x1)
lin_transition = np.vectorize(_lin_transition)

def rotate2D(X, Y, angle, origin=[0,0]):
    """Clockwise rotation about origin in 2D"""
    # angle in degrees
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    R = np.array(((c, -s), (s, c)))

    xo, yo = origin

    Xr = xo + c * (X - xo) + s * (Y - yo)
    Yr = yo - s * (X - xo) + c * (Y - yo)

    return Xr, Yr

def mean_filter(x, n=1):
    # average the 1d numpy array x with the values of the nearest neighbors
    # at distance n. n=0: no averaging, n=1: x[i] = (x[i-1] + x[i] + x[i+1])/3
    x_f = np.copy(x)

    for i in range(len(x)):
        for j in np.arange(n)+1:
            x_f[i] += x[min(i+j, len(x)-1)] + x[max(i-j,0)]
        x_f[i] /= (2*n+1)
    return x_f

class GaussianBeam():

    def __init__(self, lam, w0):

        self.w0 = w0 # beam waist radius
        self.lam = lam # wavelength
        self.zR = pi * w0**2 / self.lam # rayleigh distance

    def w(self, z):
        """ Returns 1/e radius at distance z from waist """
        return self.w0 * np.sqrt(1 + (z/self.zR)**2)

    def R(self, z):
        """
        Returns radius of curvature of wavefront at
        distance z from waist
        """

        # prevent divergence at z=0
        if type(z)==np.ndarray:
            z[z==0.0] = 1e-6
        else:
            if z==0:
                z = 1e-6

        return z * (1 + (self.zR/z)**2)

    def gouy_ph(self, z):
        """ Gouy phase """
        return np.arctan(z/self.zR)

    def ampl(self, z, r):
        """ Max amplitude at distance z from waist and at a radius r """
        return self.w0 / self.w(z) * np.exp ( - (r/self.w(z))**2)

    def phase(self, z, r):
        """ Phase at distance z from waist and at a radius r """
        return 2*pi/self.lam * (z + r**2/2/self.R(z)) - self.gouy_ph(z)

    def E(self, z, r):
        """ Electric field amplitude at distance z from waist and at a radius r """
        return self.ampl(z, r) * np.cos(self.phase(z,r))

    def test():

        from scipy import constants as co

        f0 = 50e9
        lam = co.c / f0
        a = 0.7 # minor radius
        w0 = 4.7 * 1e-3
        foc = 120 * 1e-3 # focal length

        w0, lam, foc = w0/a, lam/a, foc/a

        r_focal = 0.5 # radial position of the cutoff (norm. by a)
        r_ant = r_focal + foc # radial pos antenna (norm. by a)

        gauss_beam = GaussianBeam(lam, w0)

        xmi, xma, ymi, yma = 0.0, r_ant, -0.2, 0.2
        extent = [xmi, xma, ymi, yma]

        x = np.linspace(xmi, xma, 1000)
        y = np.linspace(ymi, yma, 1000)
        X, Y = np.meshgrid(x,y)
        angle = 15
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        R = np.array(((c, -s), (s, c)))

        Xr = r_focal + c * (X - r_focal) + s * Y
        Yr = -s * (X - r_focal) + c * Y
        #
        #
        #
        # print(R)
        #
        # [xpr, ypr] = np.dot(R, [x, y])
        # [xpr, ypr] = np.dot(R, np.array([X, Y]))
        #
        # Xpr, Ypr = np.meshgrid(xpr,ypr)
        #

        E = gauss_beam.E(Xr - r_ant, Yr)

        fig, ax = plt.subplots()
        from fullwave2d.visualize import get_default_imstyle
        imst = get_default_imstyle()
        imst.imshow_kwargs['extent'] = extent
        imst.get_im(E, ax)

        theta  = np.linspace(0, 2*pi, 100)
        ax.autoscale(False)
        ax.plot(r_focal * np.sin(theta),  r_focal * np.cos(theta), 'b--')
        ax.plot(r_focal + (r_ant - r_focal) * np.sin(theta),  (r_ant-r_focal) * np.cos(theta), 'r--')
        # ax.plot(theta, r_ant * np.sin(theta), 'r--')
        plt.show()



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    GaussianBeam.test()
