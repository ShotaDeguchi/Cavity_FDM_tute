"""
********************************************************************************
2D lid-driven cavity flow
********************************************************************************
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import spatial_op_2D as sp_op
from cavity_ref import Ghia, Erturk


parser = argparse.ArgumentParser()
parser.add_argument("--Nx", type=int, default=101, help="resolution")
parser.add_argument("--Lx", type=float, default=1., help="domain size")
parser.add_argument("--Ux", type=float, default=1., help="lid velocity")
parser.add_argument("--rho", type=float, default=1., help="density")
parser.add_argument("--nu", type=float, default=1e-2, help="kinematic viscosity")
parser.add_argument("--Cs", type=float, default=0., help="Smagorinsky constant (typically 0.1)")
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
    Nx, Ny = args.Nx, args.Nx
    Lx, Ly = args.Lx, args.Lx
    Ux, Uy = args.Ux, 0.
    rho, nu = args.rho, args.nu
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
    x = np.linspace(0., Lx, Nx)
    y = np.linspace(0., Ly, Ny)
    x, y = np.meshgrid(x, y, indexing="ij")
    u = np.zeros((Nx, Ny)) + 1e-6
    v = np.zeros((Nx, Ny)) + 1e-6
    p = np.zeros((Nx, Ny)) + 1e-6
    b = np.zeros((Nx, Ny)) + 1e-6

    # time
    dim = 2
    dt_c = dx / (Ux * dim)
    dt_d = 1. / 2. * dx**2 / (nu * dim)
    dt = min(dt_c, dt_d)
    dt *= .4   # safety
    C = Ux * dt / dx
    D = nu * dt / dx**2
    Re = Ux * Lx / nu
    print(f"dx: {dx:.3e}, dt: {dt:.3e}, C: {C:.3e}, D: {D:.3e}, Re: {Re:.3e}")

    # reference
    Ghia_dict = Ghia(int(Re))
    # Erturk_dict = Erturk(Re)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.scatter(Ghia_dict["u"], Ghia_dict["y"], color="k", marker="X", label="Ghia")
    # ax.scatter(Erturk_dict["u"], Erturk_dict["y"], label="Erturk")
    ax.set(
        xlim=(-.3, 1.1),
        ylim=(-.1, 1.1),
        xlabel=r"$u$",
        ylabel=r"$y$",
        title=rf"Horizontal velocity"
    )
    ax.legend()

    ax = fig.add_subplot(122)
    ax.scatter(Ghia_dict["x"], Ghia_dict["v"], color="k", marker="X", label="Ghia")
    # ax.scatter(Erturk_dict["x"], Erturk_dict["v"], label="Erturk")
    ax.set(
        xlim=(-.1, 1.1),
        ylim=(-.3, .3),
        xlabel=r"$x$",
        ylabel=r"$v$",
        title=rf"Vertical velocity"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(path_fig / "reference.png")
    plt.close()

    # main
    t = 0.
    T = 60.
    it = 0
    maxiter_ppe = int(1e4)
    tol_ppe = 1e-6
    tol_vel = 1e-6
    while t < T:
        # update
        t += dt
        it += 1

        # previous velocity
        u_old = np.copy(u)
        v_old = np.copy(v)

        # intermediate velocity
        u_hat = np.copy(u)
        v_hat = np.copy(v)

        # advection
        advc_x, advc_y = sp_op.advection(
            u_old, v_old,
            u_old, v_old,
            dx, dy, dt, scheme="1st"
        )

        # diffusion
        diff_x, diff_y = sp_op.diffusion(
            nu,
            u_old, v_old,
            dx, dy, scheme="2nd", Cs=args.Cs
        )

        # intermediate velocity
        u_hat[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * (- advc_x + diff_x)
        v_hat[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * (- advc_y + diff_y)

        # PPE
        div_hat = sp_op.divergence(u_hat, v_hat, dx, dy)
        b[1:-1, 1:-1] = rho / dt * div_hat
        for it_ppe in range(0, maxiter_ppe+1):
            p_old = np.copy(p)
            p[1:-1, 1:-1] = 1. / (2. * (dx**2 + dy**2)) \
                            * (
                                - b[1:-1, 1:-1] * dx**2 * dy**2 \
                                + (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dy**2 \
                                + (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dx**2
                            )
            p[:, -1] = p[:, -2]   # North
            p[:,  0] = p[:,  1]   # South
            p[-1, :] = p[-2, :]   # East
            p[0,  :] = p[1,  :]   # West
            p[1,  1] = 0.         # bottom left corner

            # convergence
            p_flatten = p.flatten()
            p_old_flatten = p_old.flatten()
            res_ppe = np.linalg.norm(p_flatten - p_old_flatten, 2) / np.linalg.norm(p_flatten, 2)
            if it_ppe % int(maxiter_ppe / 5) == 0:
                print(f"    [PPE] it_ppe: {it_ppe:06d}, res_ppe: {res_ppe:.3e}")
            if res_ppe < tol_ppe:
                print(f"    [PPE] converged")
                break

        # pressure correction
        p_x = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2. * dx)
        p_y = (p[1:-1, 2:] - p[1:-1, :-2]) / (2. * dy)

        u[1:-1, 1:-1] = u_hat[1:-1, 1:-1] + dt * (- p_x / rho)
        v[1:-1, 1:-1] = v_hat[1:-1, 1:-1] + dt * (- p_y / rho)

        # boundary
        u[:, -1], v[:, -1] = Ux, 0.   # North
        u[:,  0], v[:,  0] = 0., 0.   # South
        u[-1, :], v[-1, :] = 0., 0.   # East
        u[0, :],  v[0, :]  = 0., 0.   # West

        # convergence
        C = np.max(np.abs(u)) * dt / dx
        D = nu * dt / dx**2
        Re = Ux * Lx / nu
        u_flatten = u.flatten()
        v_flatten = v.flatten()
        u_old_flatten = u_old.flatten()
        v_old_flatten = v_old.flatten()
        res_u = np.linalg.norm(u_flatten - u_old_flatten, 2) / np.linalg.norm(u_flatten, 2)
        res_v = np.linalg.norm(v_flatten - v_old_flatten, 2) / np.linalg.norm(v_flatten, 2)
        print(f"\n********")
        print(f"[MAIN] it: {it:06d}, t: {t:.3f}/{T:.3f}, dx: {dx:.3e}, dt: {dt:.3e}")
        print(f"[MAIN] C: {C:.3e}, D: {D:.3e}, Re: {Re:.3e}")
        print(f"[MAIN] res_u: {res_u:.3e}, res_v: {res_v:.3e}")
        print(f"********\n")
        if res_u < tol_vel and res_v < tol_vel:
            print(f"[MAIN] converged")
            break

        # save
        if it % 1000 == 0:
            print(f"saving: {it:06d}")
            np.savez(path_npz / f"res_{it:06d}.npz", x=x, y=y, u=u, v=v, p=p)

            fig = plt.figure(figsize=(12, 5))

            ax = fig.add_subplot(121)
            vel_norm = np.sqrt(u**2 + v**2)
            vmin, vmax = 0., Ux
            # vmin, vmax = vel_norm.min(), vel_norm.max()
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            cont = ax.contourf(x, y, vel_norm, levels=levels, cmap="turbo", extend="both")
            fig.colorbar(cont, ax=ax, ticks=ticks)
            u_norm, v_norm = u / vel_norm, v / vel_norm
            plt.quiver(
                x[::Nx//20,::Ny//20], y[::Nx//20,::Ny//20],
                u_norm[::Nx//20,::Ny//20], v_norm[::Nx//20,::Ny//20],
                # vel_norm[::Nx//20,::Ny//20],
                # cmap="magma", pivot="tail", clim=(vmin, vmax),
                color="w", pivot="tail"
            )
            # vel_hat_norm = np.sqrt(u_hat**2 + v_hat**2)
            # u_hat_norm, v_hat_norm = u_hat / vel_hat_norm, v_hat / vel_hat_norm
            # plt.quiver(
            #     x[::Nx//20,::Ny//20], y[::Nx//20,::Ny//20],
            #     u_hat_norm[::Nx//20,::Ny//20], v_hat_norm[::Nx//20,::Ny//20],
            #     # vel_hat_norm[::Nx//20,::Ny//20],
            #     # cmap="magma", pivot="tail", clim=(vmin, vmax),
            #     color="r", pivot="tail"
            # )
            ax.set(
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"Velocity norm at $t: {t:.3f}$"
            )

            ax = fig.add_subplot(122)
            p_bar = p - np.mean(p)
            vmin, vmax = -.1, .1
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            cont = ax.contourf(x, y, p_bar, levels=levels, cmap="seismic", extend="both")
            fig.colorbar(cont, ax=ax, ticks=ticks)
            # p_grad_norm = np.sqrt(p_x**2 + p_y**2)
            # p_x_norm, p_y_norm = p_x / (rho * p_grad_norm), p_y / (rho * p_grad_norm)
            # plt.quiver(
            #     x[1:-1:Nx//20,1:-1:Ny//20], y[1:-1:Nx//20,1:-1:Ny//20],
            #     - p_x_norm[::Nx//20,::Ny//20], - p_y_norm[::Nx//20,::Ny//20],
            #     # p_grad_norm[::Nx//20,::Ny//20],
            #     # cmap="magma", pivot="tail", clim=(vmin, vmax),
            #     color="w", pivot="tail"
            # )
            ax.set(
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"Pressure (shifted) at $t: {t:.3f}$"
            )
            fig.savefig(path_fig / f"vel_prs_snapshot.png")
            fig.savefig(path_fig / f"vel_prs_{it:06d}.png")
            plt.close()

            # comparison
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(121)
            ax.scatter(Ghia_dict["u"], Ghia_dict["y"], color="k", marker="X", label="Ghia")
            # ax.scatter(Erturk_dict["u"], Erturk_dict["y"], label="Erturk")
            ax.plot(u[Nx//2, :], y[Ny//2, :], color="r", ls="--", label="FDM")
            ax.set(
                xlim=(-.3, 1.1),
                ylim=(-.1, 1.1),
                xlabel=r"$u$",
                ylabel=r"$y$",
                title=rf"Horizontal velocity"
            )
            ax.legend(loc="lower right")
            ax = fig.add_subplot(122)
            ax.scatter(Ghia_dict["x"], Ghia_dict["v"], color="k", marker="X", label="Ghia")
            # ax.scatter(Erturk_dict["x"], Erturk_dict["v"], label="Erturk")
            ax.plot(x[:, Ny//2], v[:, Ny//2], color="r", ls="--", label="FDM")
            ax.set(
                xlim=(-.1, 1.1),
                ylim=(-.3, .2),
                xlabel=r"$x$",
                ylabel=r"$v$",
                title=rf"Vertical velocity"
            )
            ax.legend(loc="lower left")
            plt.tight_layout()
            fig.savefig(path_fig / f"comparison_snapshot.png")
            fig.savefig(path_fig / f"comparison_{it:06d}.png")
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
