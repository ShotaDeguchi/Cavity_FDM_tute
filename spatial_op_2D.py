"""
********************************************************************************
utilities (2D spatial operators)
********************************************************************************
"""

import numpy as np


def advection(u, v, f, g, dx, dy, dt, scheme="1st"):
    """
    advection of f and g by u and v
    """
    if scheme == "1st":
        # 1st derivative
        f_x = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2. * dx)
        f_y = (f[1:-1, 2:] - f[1:-1, :-2]) / (2. * dy)
        g_x = (g[2:, 1:-1] - g[:-2, 1:-1]) / (2. * dx)
        g_y = (g[1:-1, 2:] - g[1:-1, :-2]) / (2. * dy)

        # 2nd derivative
        f_xx = (f[2:, 1:-1] - 2. * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
        f_yy = (f[1:-1, 2:] - 2. * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
        g_xx = (g[2:, 1:-1] - 2. * g[1:-1, 1:-1] + g[:-2, 1:-1]) / dx**2
        g_yy = (g[1:-1, 2:] - 2. * g[1:-1, 1:-1] + g[1:-1, :-2]) / dy**2

        # advection
        advc_x = (u[1:-1, 1:-1] * f_x - np.abs(u[1:-1, 1:-1]) * dx / 2. * f_xx) \
                + (v[1:-1, 1:-1] * f_y - np.abs(v[1:-1, 1:-1]) * dy / 2. * f_yy)
        advc_y = (u[1:-1, 1:-1] * g_x - np.abs(u[1:-1, 1:-1]) * dx / 2. * g_xx) \
                + (v[1:-1, 1:-1] * g_y - np.abs(v[1:-1, 1:-1]) * dy / 2. * g_yy)

    elif scheme == "LW":
        # 1st order derivatives
        f_x = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2. * dx)
        f_y = (f[1:-1, 2:] - f[1:-1, :-2]) / (2. * dy)
        g_x = (g[2:, 1:-1] - g[:-2, 1:-1]) / (2. * dx)
        g_y = (g[1:-1, 2:] - g[1:-1, :-2]) / (2. * dy)

        # 2nd order derivatives
        f_xx = (f[2:, 1:-1] - 2. * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
        f_yy = (f[1:-1, 2:] - 2. * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
        g_xx = (g[2:, 1:-1] - 2. * g[1:-1, 1:-1] + g[:-2, 1:-1]) / dx**2
        g_yy = (g[1:-1, 2:] - 2. * g[1:-1, 1:-1] + g[1:-1, :-2]) / dy**2

        # advection
        advc_x = (u[1:-1, 1:-1] * f_x - np.abs(u[1:-1, 1:-1])**2 * dt / 2. * f_xx) \
                + (v[1:-1, 1:-1] * f_y - np.abs(v[1:-1, 1:-1])**2 * dt / 2. * f_yy)
        advc_y = (u[1:-1, 1:-1] * g_x - np.abs(u[1:-1, 1:-1])**2 * dt / 2. * g_xx) \
                + (v[1:-1, 1:-1] * g_y - np.abs(v[1:-1, 1:-1])**2 * dt / 2. * g_yy)

    elif scheme == "QUICK":
        # need longer stencil
        raise NotImplementedError

    elif scheme == "QUICKEST":
        # need longer stencil
        raise NotImplementedError

    elif scheme == "KK":
        # need longer stencil
        raise NotImplementedError

    return advc_x, advc_y


def diffusion(nu, f, g, dx, dy, scheme="2nd", Cs=.1):
    """
    diffusion of f and g with nu
    """
    # strain rate tensor
    f_x = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2. * dx)
    f_y = (f[1:-1, 2:] - f[1:-1, :-2]) / (2. * dy)
    g_x = (g[2:, 1:-1] - g[:-2, 1:-1]) / (2. * dx)
    g_y = (g[1:-1, 2:] - g[1:-1, :-2]) / (2. * dy)

    S11, S12 = f_x, .5 * (f_y + g_x)
    S21, S22 = S12, g_y
    S = np.sqrt(2. * (S11**2 + S22**2 + 2. * S12**2))

    # Smagorinsky model
    l0 = Cs * dx
    nu_t = l0**2 * S
    nu += nu_t
    print(f"  [LES] Smagorinsky: Cs: {Cs:.3f}, l0: {l0:.3e}, nu_t.min: {nu_t.min():.3e}, nu_t.max: {nu_t.max():.3e}")

    if scheme == "2nd":
        # 2nd order accurate, 2nd order derivatives
        f_xx = (f[2:, 1:-1] - 2. * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
        f_yy = (f[1:-1, 2:] - 2. * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
        g_xx = (g[2:, 1:-1] - 2. * g[1:-1, 1:-1] + g[:-2, 1:-1]) / dx**2
        g_yy = (g[1:-1, 2:] - 2. * g[1:-1, 1:-1] + g[1:-1, :-2]) / dy**2

    elif scheme == "4th":
        # need longer stencil
        raise NotImplementedError

    # laplacian
    lap_f = f_xx + f_yy
    lap_g = g_xx + g_yy

    # diffusion
    diff_x = nu * lap_f
    diff_y = nu * lap_g
    return diff_x, diff_y


def divergence(u, v, dx, dy):
    """
    divergence of u and v
    """
    u_x = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dx)
    v_y = (v[1:-1, 2:] - v[1:-1, :-2]) / (2. * dy)
    div = u_x + v_y
    return div


def gradient(p, dx, dy):
    """
    gradient of p
    """
    p_x = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2. * dx)
    p_y = (p[1:-1, 2:] - p[1:-1, :-2]) / (2. * dy)
    return p_x, p_y


def voriticity(u, v, dx, dy):
    """
    voriticity of u and v
    """
    u_y = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dy)
    v_x = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2. * dx)
    zeta = v_x - u_y
    return zeta
