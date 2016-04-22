import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

maxIter = 100000
Re = 100
nx = 500
ny = 150
q = 9
r = ny / 3
uLB = 0.001
nulb = uLB * 2 * r / Re
tau = 1.0 / (3. * nulb + 0.5)

e_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

c_sqrd = 1 / 3

rho = np.ones((nx, ny))

uc = np.zeros((nx, ny, q))
obstacle_x = nx / 2
obstacle_y = ny / 2
obstacle_r = ny / 7





def equilibrium(rho, ux, uy):  # Equilibrium distribution function.
    feq = np.zeros((nx, ny, q))
    for i in range(q):
        uc[:,:,i] = ux * cx[i] + uy * cy[i]
    uc2 = uc ** 2
    u2 = ux ** 2 + uy ** 2

    for i in range(q):
        feq[:,:,i] = rho * w_i[i] * (1 + uc[:,:,i] / c_sqrd + (uc2[:,:,i] - c_sqrd * u2) / (2*c_sqrd **2))

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
        ux[:,y] = u_max * (1 - np.abs(1 - 2*y/max_y))
    return ux, uy

def poiseulle_flow_boundary(max_y, u_max):
    ux = np.zeros((max_y))
    for y in range(max_y):
        ux[y] = (u_max / (ny /2 ) ** 2) * (1 - (np.abs(y-ny/2) / (ny / 2)) ** 2)
    return ux

boundary = set_boundary(nx, ny, obstacle_x, obstacle_y, obstacle_r)
not_boundary = boundary == False

BB = np.zeros((nx, ny, q))
# ux, uy = poiseuille_flow_channel(nx, ny, uLB)
ux = np.zeros((nx, ny))
uy = np.zeros((nx, ny))
ux[boundary] = 0
uy[boundary] = 0
f_i = equilibrium(rho, ux, uy)
f_eq = np.copy(f_i)
f_out = np.copy(f_i)

p_flow_bdry = poiseulle_flow_boundary(ny, uLB)
p_flow_channel = poiseuille_flow_channel(nx, ny, uLB)


for t in range(maxIter):
    for i in range(q):
        f_i[:, :, i] = np.roll(np.roll(f_out[:, :, i], cx[i], axis=0), cy[i], axis=1)
    f_i[0] = f_i[1] # Taking away periodic boundary conditions on the inflow
    f_i[-1] = f_i[-2] # And also at the outflow

    for i in range(q):
        f_i[boundary, i] = f_i[boundary, opp[i]]

    ux[0,:] = p_flow_bdry
    for i in range(nx):
        ux[i,not_boundary[i,:]] += 1.0 * ux[0, not_boundary[i,:]]

    rho = np.sum(f_i, axis=2)
    f_eq = equilibrium(rho, ux, uy)
    f_out = f_i - (f_i - f_eq) / tau



    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho

    # print(np.dot(f_i, cx))

    if (t % 50 == 0):
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



