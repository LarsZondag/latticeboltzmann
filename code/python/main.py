import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

# Flow constants
maxIter = 100000        # amount of cycles
Re = 220                # Reynolds number
nx = 520
ny = 180
q = 9                   # number of possible directions
uLB = 0.04              # maximum velocity of Poiseuille flow
obstacle_x = nx / 4     # x location of the cylinder
obstacle_y = ny / 2     # y location of the cylinder
obstacle_r = ny / 9     # radius of the cylinder
nulb = uLB * obstacle_r / Re    # kinematic viscosity
tau = (3. * nulb + 0.5) # relaxation parameter
cylinder = True

# D2Q9 Lattice constants
# e_i gives the directions of all 9 velocity vectors.
# opp gives the indices that correspond to the index of e_i containing the opposite velocity vector
e_i = np.zeros((q, 2), dtype=np.int64)
opp = np.zeros((q, 1), dtype=np.int8)
e_i[0] = [ 0,  0];      opp[0] = 0
e_i[1] = [ 1,  0];      opp[1] = 3
e_i[2] = [ 0,  1];      opp[2] = 4
e_i[3] = [-1,  0];      opp[3] = 1
e_i[4] = [ 0, -1];      opp[4] = 2
e_i[5] = [ 1,  1];      opp[5] = 7
e_i[6] = [-1,  1];      opp[6] = 8
e_i[7] = [-1, -1];      opp[7] = 5
e_i[8] = [ 1, -1];      opp[8] = 6
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])


## Functions: ##
# Equilibrium distribution function.
@jit
def equilibrium(rho, max_x, max_y, directions, velo_x, velo_y, c):
    feq = np.zeros((max_x, max_y, directions))
    uc = np.zeros((max_x, max_y, directions))
    for i in range(directions):
        uc[:,:,i] = velo_x * c[i, 0] + velo_y * c[i, 1]
    uc *= 3
    uc2 = uc ** 2
    u2 = 3 / 2 * (velo_x ** 2 + velo_y ** 2)

    for i in range(directions):
        feq[:,:,i] = rho * w_i[i] * (1 + uc[:,:,i] + 0.5 * uc2[:,:,i] - u2)

    return feq

# Create a boundary mask
def set_boundary(max_x, max_y, obst_x, obst_y, obst_r, cylinder):
    b = np.zeros((max_x, max_y))
    if cylinder:
        for i in range(max_x):
            for j in range(max_y):
                b[i, j] = (i - obst_x) ** 2 + (j - obst_y) ** 2 <= obst_r ** 2
    b[:, 0] = True
    b[:, -1] = True
    return b == 1

# Initial condition
def poiseuille_flow_channel(max_x, max_y, u_max):
    uy = np.zeros((max_x, max_y))
    ux = np.zeros((max_x, max_y))
    for y in range(max_y):
        ux[:,y] = (u_max / (max_y /2 ) ** 2) * (1 - (np.abs(y-max_y/2) / (max_y / 2)) ** 2)
    return ux, uy


# Initialize the boundary once. The boundary depends on the box's dimensions and the object being placed
boundary = set_boundary(nx, ny, obstacle_x, obstacle_y, obstacle_r, cylinder)

# Initialize the poiseulle flow velocity profile once to reference later
p_flow_channel_x, p_flow_channel_y = poiseuille_flow_channel(nx, ny, uLB)

# Conditions to start the simulation with:
# Starting velocity profile can be set here. We chose a poiseulle flow profile
ux = p_flow_channel_x
uy = p_flow_channel_y

# Make sure there is not flow on the boundary
ux[boundary] = 0
uy[boundary] = 0
rho = np.ones((nx, ny))

# The velocity profile is incorporated into the equilibrium distribution.
f_i = equilibrium(rho, nx, ny, q, ux, uy, e_i)
f_eq = np.copy(f_i)

## Main loop ##
for t in range(maxIter):
    # Right side boundary condition to make sure the flow doesn't go backwards
    f_i[ -1, :, [3, 6, 7]] = f_i[ -2, :, [3, 6, 7]]

    # Calculate density and velocities
    rho = np.sum(f_i, axis=2)
    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    # Increment velocities in x-direction to mimic a constant pressure drop
    ux += 0.7 * uLB

    # Calculate the equilibrium distribution based on the density and velocities
    f_eq = equilibrium(rho, nx, ny, q, ux, uy, e_i)

    # Collision / relaxation step
    f_out = f_i - (f_i - f_eq) / tau

    # Bounce-back boundary conditions. This step flips the directions of the velocities of all points lying inside objects.
    for i in range(q):
        f_out[boundary, i] = f_i[boundary, opp[i]]

    # Streaming step
    for i in range(q):
        f_i[:, :, i] = np.roll(np.roll(f_out[:, :, i], e_i[i, 0], axis=0), e_i[i, 1], axis=1)

    # Make images of velocity in the x and y direction and of the speed
    if (t % 100 == 0):
        plt.clf()
        plt.imshow(ux.transpose(), cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("velx/" + str(t / 100) + ".png")

        plt.clf()
        plt.imshow(uy.transpose(), cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("vely/" + str(t / 100) + ".png")

        plt.clf()
        plt.imshow((np.sqrt(ux ** 2 + uy ** 2)).transpose(), cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("vel/" + str(t / 100) + ".png")



