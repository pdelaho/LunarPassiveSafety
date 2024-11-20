import matplotlib.pyplot as plt
import numpy as np


def plot_target_traj_syn(target_traj, L_point, mu):
    """Takes in the trajectory of the target spacecraft in the synodic frame, the considered Lagrange point, and the mass ratio parameter,
    plots the given trajectory in the synodic frame.

    Args:
        target_traj (n_timex6 vector): trajectory of the target spacecraft in synodic frame
        L_point (float): coordinate of the Lagrange point along the x axis in synodic frame
        mu (float): mass ratio parameter in 3-body problem
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], color='r', label="Target's orbit")
    ax.scatter(0, 0, 0, label='Moon')
    ax.scatter(L_point - (1 - mu), 0, 0, label='L2')
    ax.scatter(target_traj[0, 0],target_traj[0, 1], target_traj[0, 2], label='Start')
    ax.axis('equal')
    ax.set_xlabel('X [nd]')
    ax.set_ylabel('Y [nd]')
    ax.set_zlabel('Z [nd]')
    ax.legend()
    plt.title("Target's orbit in the synodic frame")
    plt.grid()


def plot_chaser_traj_lvlh(chaser_traj, LU):
    """Takes in the trajectory of the chaser spacecraft in the LVLH frame and the length parameter, plots the given trajectory rescaled to km.

    Args:
        chaser_traj (n_timex6 vector): trajectory of the chaser spacecraft in LVLH frame
        LU (float): length parameter in 3-body problem
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(chaser_traj[:, 0] * LU, - chaser_traj[:, 1] * LU, - chaser_traj[:, 2] * LU, color='r', label="Chaser's trajectory")
    ax.scatter(chaser_traj[0, 0] * LU, - chaser_traj[0, 1] * LU, - chaser_traj[0, 2] * LU, label='Start')
    ax.scatter(chaser_traj[-1, 0] * LU, - chaser_traj[-1, 1] * LU, - chaser_traj[-1, 2] * LU, label='End')
    ax.axis('equal')
    
    # LVLH [i,j,k] = [T, -N, -R]
    ax.set_xlabel('T [km]')
    ax.set_ylabel('N [km]')
    ax.set_zlabel('R [km]')
    ax.legend()
    plt.title("Chaser's trajectory in the LVLH frame")
    plt.grid()


def plot_chaser_traj_lvlh_scvx(chaser_traj, ax, LU, color, lineWidth=1):
    """Takes in the trajectory of the chaser spacecraft in the LVLH frame, the figure on which to plot, the length parameter,
    the color for the plot and the line width we'd like, plots the trajectory of the chaser spacecraft in the LVLH frame rescaled to km.

    Args:
        chaser_traj (n_timex6): trajectory of the chaser spacecraft in LVLH frame
        ax (figure): figure on which we want to plot the trajectory
        LU (float): length parameter in 3-body problem
        color (color): color of the trajectory
        lineWidth (int, optional): line width to plot the trajectory. Defaults to 1.
    """
    
    ax.plot(chaser_traj[:, 0] * LU, - chaser_traj[:, 1] * LU, - chaser_traj[:, 2] * LU, color=color, label="Chaser's trajectory", linewidth=lineWidth)
    ax.scatter(chaser_traj[0, 0] * LU, - chaser_traj[0, 1] * LU, - chaser_traj[0, 2] * LU, label='Start')
    ax.scatter(chaser_traj[-1, 0] * LU, - chaser_traj[-1, 1] * LU, - chaser_traj[-1, 2] * LU, label='End')
    ax.axis('equal')
    
    # LVLH [i,j,k] = [T, -N, -R]
    ax.set_xlabel('T [km]')
    ax.set_ylabel('N [km]')
    ax.set_zlabel('R [km]')
    ax.legend()
    plt.title("Chaser's trajectory in the LVLH frame")
    plt.grid()


def plot_chaser_traj_lvlh_check(chaser_traj, ax, LU, color, lineWidth=1):
    """Takes in the trajectory of the chaser spacecraft in the LVLH frame, the figure on which to plot, the length parameter,
    the color for the plot and the line width we'd like, plots the trajectory of the chaser spacecraft in the LVLH frame rescaled to km.

    Args:
        chaser_traj (n_timex6): trajectory of the chaser spacecraft in LVLH frame
        ax (figure): figure on which we want to plot the trajectory
        LU (float): length parameter in 3-body problem
        color (color): color of the trajectory
        lineWidth (int, optional): line width to plot the trajectory. Defaults to 1.
    """
    
    ax.plot(chaser_traj[:, 0] * LU, - chaser_traj[:, 1] * LU, - chaser_traj[:, 2] * LU, color=color, linewidth=lineWidth)
    ax.axis('equal')
    
    # LVLH [i,j,k] = [T, -N, -R]
    ax.set_xlabel('T [km]')
    ax.set_ylabel('N [km]')
    ax.set_zlabel('R [km]')
    ax.legend()


def analysis(chaser_nonlin_traj, chaser_lin_traj, n_time):
    """Takes in the trajectory for the chaser spacecraft, one being using regular non-linear dynamics, the other using the linearized
    dynamics, and th enumber of time steps, returns vectors computing the error in position and in velocity between the 2.

    Args:
        chaser_nonlin_traj (n_timex6 vector): trajectory of the chaser spacecraft using non-linear dynamics
        chaser_lin_traj (n_timex6 vector): trajectory of the chaser spacecraft using linearized dynamics
        n_time (int): number of timesteps in each vectors

    Returns:
        vector(n_timex1 vector): contains the error in position at each time step
        vector(n_timex1 vector): contains the error in velocity at each time step
    """
    
    error_lin_pos = np.empty(n_time)
    error_lin_vel = np.empty(n_time)
    
    for i in range(n_time):
        error_lin_pos[i] = np.linalg.norm(chaser_nonlin_traj[i, :3] - chaser_lin_traj[i, :3])
        error_lin_vel[i] = np.linalg.norm(chaser_nonlin_traj[i, 3:6] - chaser_lin_traj[i, 3:6])
    
    return error_lin_pos, error_lin_vel


def plot_ellipse_2D(P_inv, ax):
    """Takes in the inverse of the shape matrix of an ellipse and a figure on which to plot it, plots the corresponding 2D ellipse.
    Taken from Yuji's code.

    Args:
        P_inv (2x2 matrix): inverse of the shape matrix of an ellipse
        ax (figure): figure on which to plot the corresponding ellipse
    """
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(P_inv)

    # Length of semi-axes (sqrt of eigenvalues)
    a = np.sqrt(1 / eigenvalues[0])
    b = np.sqrt(1 / eigenvalues[1])

    # Angle for parametric plot
    t = np.linspace(0, 2 * np.pi, 100)

    # Parametric equations of the ellipse
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)
    
    # Rotate the ellipse to align with eigenvectors
    ellipse = np.array([ellipse_x, ellipse_y])
    rotated_ellipse = eigenvectors @ ellipse

    # Plot
    ax.plot(rotated_ellipse[0, :], rotated_ellipse[1, :])
    
def plot_ellipse_3D(P_inv, ax, LU, TU, label, color, type='pos'):
    """Takes in the inverse of the shape matrix of an ellipsoid, a figure on which to plot it, the length and time parameters,
    what we want to label it, the color, and the part of the ellipsoid we want to plot.

    Args:
        P_inv (6x6 matrix): inverse of the ellipsoid's shape matrix
        ax (figure): the figure on which to plot it
        LU (float): length parameter in 3-body problem
        TU (float): time parameter in 3-body problem
        label (string): what we want to label the ellipsoid
        color (color): which color to plot the ellipsoid
        type (str, optional): pos or vel depending on the part of the ellipsoid to plot. Defaults to 'pos'.
    """
    center = [0, 0, 0]

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(P_inv)
    radii = 1.0 / np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    if type == 'pos':
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) * LU
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) * LU
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) * LU
        # ax.set_xlabel('T [nd]')
        # ax.set_ylabel('N [nd]')
        # ax.set_zlabel('R [nd]')
    if type == 'vel':
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) # * LU/TU
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) # * LU/TU
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) # * LU/TU
        # ax.set_xlabel('T [nd]')
        # ax.set_ylabel('N [nd]')
        # ax.set_zlabel('R [nd]')
    
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], rotation) + center

    # ax.set_xlabel('T [km]')
    # ax.set_ylabel('N [km]')
    # ax.set_zlabel('R [km]')
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, alpha=0.4, label = label, color = color) # color = 'b'
    

def plot_ellipse_3D_scvx(P_inv, ax, LU, TU, label, color, type='pos'):
    """Takes in the inverse of the shape matrix of an ellipsoid, a figure on which to plot it, the length and time parameters,
    what we want to label it, the color, and the part of the ellipsoid we want to plot.

    Args:
        P_inv (6x6 matrix): inverse of the ellipsoid's shape matrix
        ax (figure): the figure on which to plot it
        LU (float): length parameter in 3-body problem
        TU (float): time parameter in 3-body problem
        label (string): what we want to label the ellipsoid
        color (color): which color to plot the ellipsoid
        type (str, optional): pos or vel depending on the part of the ellipsoid to plot. Defaults to 'pos'.
    """
    center = [0, 0, 0]

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(P_inv)
    radii = 1.0 / np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    if type == 'pos':
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) * LU
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) * LU
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) * LU
        # ax.set_xlabel('T [nd]')
        # ax.set_ylabel('N [nd]')
        # ax.set_zlabel('R [nd]')
    if type == 'vel':
        x = radii[3] * np.outer(np.cos(u), np.sin(v)) # * LU/TU
        y = radii[4] * np.outer(np.sin(u), np.sin(v)) # * LU/TU
        z = radii[5] * np.outer(np.ones_like(u), np.cos(v)) # * LU/TU
        ax.set_xlabel('T [nd]')
        ax.set_ylabel('N [nd]')
        ax.set_zlabel('R [nd]')
    
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], rotation[:3,:3]) + center

    ax.set_xlabel('T [km]')
    ax.set_ylabel('N [km]')
    ax.set_zlabel('R [km]')
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, alpha=0.4, label = label, color = color) # color = 'b'