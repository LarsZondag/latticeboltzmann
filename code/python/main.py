import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

maxIter = 10000
Re = 200
nx = 500
ny = 70
q = 9
r = ny / 9
uLB = 0.1
nulb = uLB * r / Re
tau = 1.0 / (3. * nulb + 0.5)

e_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

c_sqrd = 1 / 3

rho = np.ones((nx, ny))

f_i = np.zeros((nx, ny, q))
uc = np.zeros((nx, ny, q))


f_i[:, :, 0] = w_i[0] * rho
f_eq = np.copy(f_i)


def equilibrium(rho, ux, uy):  # Equilibrium distribution function.
    feq = np.zeros((nx, ny, q))
    for i in range(q):
        uc[:,:,i] = ux * cx[i] + uy * cy[i]
    uc2 = uc ** 2
    u2 = ux ** 2 + uy ** 2

    for i in range(q):
        feq[:,:,i] = rho * w_i[i] * (1 + uc[:,:,i] / c_sqrd + (uc2[:,:,i] - c_sqrd * u2) / (2*c_sqrd **2))

    return feq


def set_boundary(max_x, max_y, radius2):
    center = [max_x / 2, max_y / 2]
    b = np.zeros((max_x, max_y))
    for i in range(max_x):
        for j in range(max_y):
            b[i, j] = (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius2
    b[:, 0] = True
    b[:, -1] = True
    return b == 1


boundary = set_boundary(nx, ny, r ** 2)
not_boundary = boundary == False

BB = np.zeros((nx, ny, q))
ux = np.zeros((nx, ny))
uy = np.zeros((nx, ny))

for t in range(maxIter):
    f_out = f_i - (f_i - f_eq) / tau

    for i in range(q):
        f_out[boundary, i] = f_out[boundary, opp[i]]

    f_out[0,:,1] += uLB
    # f_out[-1,:,1] -= uLB

    for i in range(q):
        f_i[:, :, i] = np.roll(np.roll(f_out[:, :, i], cx[i], axis=0), cy[i], axis=1)


    rho = np.sum(f_i, axis=2)

    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho



    f_eq = equilibrium(rho, ux, uy)





    if (t % 50 == 0):
        plt.clf();
        plt.imshow((np.sqrt(ux ** 2 + uy ** 2)).transpose(), cmap=cm.cool)
        plt.savefig("vel." + str(t / 100).zfill(4) + ".png")
