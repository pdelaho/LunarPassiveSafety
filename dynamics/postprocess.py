import matplotlib.pyplot as plt
import numpy as np

def plot_target_traj_syn(target_traj, L_point, mu):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r', label="Target's orbit")
    ax.scatter(0, 0, 0, label='Moon')
    ax.scatter(L_point-(1-mu), 0, 0, label='L2')
    ax.scatter(target_traj[0,0],target_traj[0,1], target_traj[0,2], label='Start')
    ax.axis('equal')
    ax.set_xlabel('X [nd]')
    ax.set_ylabel('Y [nd]')
    ax.set_zlabel('Z [nd]')
    ax.legend()
    plt.title("Target's orbit in the synodic frame")
    plt.grid()
    
def plot_chaser_traj_lvlh(chaser_traj,LU):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(chaser_traj[:,0]*LU, -chaser_traj[:,1]*LU, -chaser_traj[:,2]*LU, color='r', label="Chaser's trajectory")
    # ax.scatter(0, 0, 0, label='Target')
    ax.scatter(chaser_traj[0,0]*LU,-chaser_traj[0,1]*LU,-chaser_traj[0,2]*LU,label='Start')
    ax.scatter(chaser_traj[-1,0]*LU,-chaser_traj[-1,1]*LU,-chaser_traj[-1,2]*LU,label='End')
    ax.axis('equal')
    # LVLH [i,j,k] = [T, -N, -R]
    ax.set_xlabel('T [km]')
    ax.set_ylabel('N [km]')
    ax.set_zlabel('R [km]')
    ax.legend()
    plt.title("Chaser's trajectory in the LVLH frame")
    plt.grid()
    
def plot_chaser_traj_lvlh_scvx(chaser_traj, ax, LU):
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax.plot(chaser_traj[:,0]*LU, -chaser_traj[:,1]*LU, -chaser_traj[:,2]*LU, color='r', label="Chaser's trajectory")
    # ax.scatter(0, 0, 0, label='Target')
    ax.scatter(chaser_traj[0,0]*LU,-chaser_traj[0,1]*LU,-chaser_traj[0,2]*LU,label='Start')
    ax.scatter(chaser_traj[-1,0]*LU,-chaser_traj[-1,1]*LU,-chaser_traj[-1,2]*LU,label='End')
    ax.axis('equal')
    # LVLH [i,j,k] = [T, -N, -R]
    ax.set_xlabel('T [km]')
    ax.set_ylabel('N [km]')
    ax.set_zlabel('R [km]')
    ax.legend()
    plt.title("Chaser's trajectory in the LVLH frame")
    plt.grid()
    
def analysis(chaser_nonlin_traj,chaser_lin_traj,n_time):
    error_lin_pos = np.empty(n_time)
    error_lin_vel = np.empty(n_time)
    
    for i in range(n_time):
        error_lin_pos[i] = np.linalg.norm(chaser_nonlin_traj[i,:3] - chaser_lin_traj[i,:3])
        error_lin_vel[i] = np.linalg.norm(chaser_nonlin_traj[i,3:6] - chaser_lin_traj[i,3:6])
    
    return error_lin_pos, error_lin_vel

# From Yuji's code
def plot_ellipse_2D(P_inv, ax):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(P_inv)

    # Length of semi-axes (sqrt of eigenvalues)
    a = np.sqrt(1/eigenvalues[0])
    b = np.sqrt(1/eigenvalues[1])
    # c = np.sqrt(1/eigenvalues[2])
    print(a, b)

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
    center = [0,0,0]

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(P_inv)
    radii = 1.0/np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    if type == 'pos':
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) * LU
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) * LU
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) * LU
        ax.set_xlabel('T [km]')
        ax.set_ylabel('N [km]')
        ax.set_zlabel('R [km]')
    if type == 'vel':
        x = radii[0] * np.outer(np.cos(u), np.sin(v)) # * LU/TU
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) # * LU/TU
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) # * LU/TU
        ax.set_xlabel('T [km/s]')
        ax.set_ylabel('N [km/s]')
        ax.set_zlabel('R [km/s]')
    
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    # ax.set_xlabel('T [km]')
    # ax.set_ylabel('N [km]')
    # ax.set_zlabel('R [km]')
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, alpha=0.2, label = label, color = color) # color = 'b'