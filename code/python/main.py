import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

maxIter = 100000
Re = 220
nx = 520
ny = 180
q = 9
uLB = 0.04
obstacle_x = nx / 4
obstacle_y = ny / 2
obstacle_r = ny / 9
nulb = uLB * obstacle_r / Re
tau = (3. * nulb + 0.5)

e_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

c_sqrd = 2 / 3

rho = np.ones((nx, ny))

uc = np.zeros((nx, ny, q))


@jit
def equilibrium(rho, ux, uy):  # Equilibrium distribution function.
    feq = np.zeros((nx, ny, q))
    uc = np.zeros((nx, ny, q))
    for i in range(q):
        uc[:,:,i] = ux * cx[i] + uy * cy[i]
    uc *= 3
    uc2 = uc ** 2
    u2 = 3 / 2 * (ux ** 2 + uy ** 2)

    for i in range(q):
        feq[:,:,i] = rho * w_i[i] * (1 + uc[:,:,i] + 0.5 * uc2[:,:,i] - u2)

    return feq


def set_boundary(max_x, max_y, obst_x, obst_y, obst_r):
    b = np.zeros((max_x, max_y))
    for i in range(max_x):
        for j in range(max_y):
            b[i, j] = (i - obst_x) ** 2 + (j - obst_y) ** 2 <= obst_r ** 2
    b[:, 0] = True
    b[:, -1] = True
    return b == 1

def poiseuille_flow_channel(max_x, max_y, u_max):
    uy = np.zeros((max_x, max_y))
    ux = np.zeros((max_x, max_y))
    for y in range(max_y):
        ux[:,y] = (u_max / (ny /2 ) ** 2) * (1 - (np.abs(y-ny/2) / (ny / 2)) ** 2)
    return ux, uy

def poiseulle_flow_boundary(max_y, u_max):
    ux = np.zeros((max_y))
    for y in range(max_y):
        ux[y] = (u_max / (ny /2 ) ** 2) * (1 - (np.abs(y-ny/2) / (ny / 2)) ** 2)
    return ux

boundary = set_boundary(nx, ny, obstacle_x, obstacle_y, obstacle_r)
not_boundary = boundary == False


p_flow_bdry = poiseulle_flow_boundary(ny, uLB)
p_flow_channel_x, p_flow_channel_y = poiseuille_flow_channel(nx, ny, uLB)

ux = p_flow_channel_x
uy = p_flow_channel_y
ux[boundary] = 0
uy[boundary] = 0
f_i = equilibrium(rho, ux, uy)
f_eq = np.copy(f_i)
f_out = np.copy(f_i)

for t in range(maxIter):
    f_i[ -1, :, [3, 6, 7]] = f_i[ -2, :, [3, 6, 7]]
    rho = np.sum(f_i, axis=2)
    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    ux += 0.7 * p_flow_channel_x
    # rho[0, :] = 1. / (1. - ux[0]) * (np.sum(f_i[0, :, [0, 2, 4]], axis=0) + 2. * np.sum(f_i[0, :, [3, 6, 7]], axis=0))
    f_eq = equilibrium(rho, ux, uy)

    f_out = f_i - (f_i - f_eq) / tau

    for i in range(q):
        f_out[boundary, i] = f_i[boundary, opp[i]]

    for i in range(q):
        f_i[:, :, i] = np.roll(np.roll(f_out[:, :, i], cx[i], axis=0), cy[i], axis=1)

    if (t % 100 == 0):
        plt.clf();
        plt.imshow((ux).transpose(), cmap=cm.Blues)
        # plt.clim(0,uLB)
        plt.colorbar()
        plt.savefig("velx/" + str(t / 100).zfill(4) + ".png")
        plt.clf();
        plt.imshow((uy).transpose(), cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("vely/" + str(t / 100).zfill(4) + ".png")
        plt.clf();
        plt.imshow((np.sqrt(ux ** 2 + uy ** 2)).transpose(), cmap=cm.Blues)
        # plt.clim(0,uLB)
        plt.colorbar()
        plt.savefig("vel/" + str(t / 100).zfill(4) + ".png")



