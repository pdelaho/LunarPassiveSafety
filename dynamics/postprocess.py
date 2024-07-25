import matplotlib.pyplot as plt

def plot_target_traj_syn(target_traj, L_point, mu):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], color='r', label="Target's orbit")
    ax.scatter(0, 0, 0, label='Moon')
    ax.scatter(L_point-(1-mu), 0, 0, label='L2')
    ax.axis('equal')
    ax.set_xlabel('X [nd]')
    ax.set_ylabel('Y [nd]')
    ax.set_zlabel('Z [nd]')
    ax.legend()
    plt.title("Target's orbit in the synodic frame")
    plt.grid()
    
def plot_chaser_traj_lvlh(chaser_traj):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(chaser_traj[:,0], chaser_traj[:,1], chaser_traj[:,2], color='r', label="Chaser's trajectory")
    ax.scatter(0, 0, 0, label='Target')
    ax.axis('equal')
    ax.set_xlabel('X [nd]')
    ax.set_ylabel('Y [nd]')
    ax.set_zlabel('Z [nd]')
    ax.legend()
    plt.title("Chaser's trajectory in the LVLH frame")
    plt.grid()
    
def analysis():
    return