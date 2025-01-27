"""
********************************************************************************
1D advection
********************************************************************************
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=101, help="resolution")
parser.add_argument("--ad", type=float, default=.4, help="advection rate")
args = parser.parse_args()


def main():
    # path
    f_name = Path(__file__).stem
    path_res = Path(f"{f_name}_res")
    path_fig = path_res / "fig"
    path_npz = path_res / "npz"
    path_fig.mkdir(parents=True, exist_ok=True)
    path_npz.mkdir(parents=True, exist_ok=True)

    # param
    nx = args.nx
    ad = args.ad
    dx = 1. / (nx - 1)
    x = np.linspace(0., 1., nx)
    u = np.zeros(nx)

    # initial condition
    u = np.where((x >= .2) & (x <= .4), 1., 0.)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, u)
    ax.set(
        xlim=(-.1, 1.1),
        ylim=(-.1, 1.1),
        xlabel=r"$x$",
        ylabel=r"$u$",
        title="initial condition",
    )
    fig.savefig(path_fig / "initial_condition.png")
    plt.close()

    # time
    T = 1.
    dt = dx / ad
    dt *= .4   # safety
    C = ad * dt / dx
    print(f"dx: {dx:.2e}, dt: {dt:.2e}, C: {C:.2f}")

    # main
    t = 0.
    it = 0
    while t < T:
        # update
        t += dt

        # advection
        it += 1
        u_old = u.copy()
        # for i in range(1, nx-1):
            # # if advection direction is known (positive)
            # u[i] = u_old[i] \
            #         + dt * (- ad * (u_old[i] - u_old[i-1]) / dx)

            # # if advection direction is unknown
            # u[i] = u_old[i] \
            #         + dt * (
            #             - ad * (u_old[i+1] - u_old[i-1]) / (2. * dx)
            #             + np.abs(ad) * dx / 2. * (u_old[i+1] - 2. * u_old[i] + u_old[i-1]) / dx**2
            #         )

        # slice operation
        u[1:-1] = u_old[1:-1] \
                    + dt * (
                        - ad * (u_old[2:] - u_old[:-2]) / (2. * dx)
                        + np.abs(ad) * dx / 2. * (u_old[2:] - 2. * u_old[1:-1] + u_old[:-2]) / dx**2
                    )

        # # Lax-Wendroff
        # u[1:-1] = u_old[1:-1] \
        #             + dt * (
        #                 - ad * (u_old[2:] - u_old[:-2]) / (2. * dx)
        #                 + np.abs(ad)**2 * dt / 2. * (u_old[2:] - 2. * u_old[1:-1] + u_old[:-2]) / dx**2
        #             )

        # boundary
        u[0] = u_old[0]
        u[-1] = u_old[-1]

        # save
        if it % 10 == 0:
            np.savez(path_npz / f"res_{it:04d}.npz", x=x, u=u)

            C = ad * dt / dx
            print(f"it: {it:04d}, t: {t:.2f}/{T:.2f}, C: {C:.2f}")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, u)
            ax.set(
                xlim=(-.1, 1.1),
                ylim=(-.1, 1.1),
                xlabel=r"$x$",
                ylabel=r"$u$",
                title=rf"$t: {t:.2f}, C: {C:.2f}$",
            )
            # ax.legend()
            fig.savefig(path_fig / f"fig_{it:04d}.png")
            plt.close()

################################################################################

def plot_setting():
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.axisbelow'] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_setting()
    main()
