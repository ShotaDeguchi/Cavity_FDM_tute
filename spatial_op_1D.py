"""
********************************************************************************
utilities (1D spatial operators)
********************************************************************************
"""

import numpy as np


def advection(ad, f, dx, dt, scheme="1st"):
    """
    advect f with advection rate ad
    """
    if scheme == "1st":
        # derivatives
        f_x = (f[2:] - f[:-2]) / (2. * dx)
        f_xx = (f[2:] - 2. * f[1:-1] + f[:-2]) / dx**2

        # advection
        advc = ad * f_x - np.abs(ad) * dx / 2. * f_xx

    elif scheme == "LW":
        # derivatives
        f_x = (f[2:] - f[:-2]) / (2. * dx)
        f_xx = (f[2:] - 2. * f[1:-1] + f[:-2]) / dx**2

        # advection
        advc = ad * f_x - np.abs(ad)**2 * dt / 2. * f_xx

    return advc


def diffusion(nu, f, dx):
    """
    diffuse f with diffusion rate nu
    """
    f_xx = (f[2:] - 2. * f[1:-1] + f[:-2]) / dx**2
    diff = nu * f_xx
    return diff
