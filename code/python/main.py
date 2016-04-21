import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

maxIter = 1000
Re = 10
nx = 100
ny = 50
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

for i in range(nx):
    for j in range(ny):
        f_i[i, j, 0] = w_i[0] * rho[i, j]

f_eq = np.copy(f_i)


def equilibrium(rho, ux, uy):  # Equilibrium distribution function.
    feq = np.zeros((nx, ny, q))
    for i in range(q):
        uc[:,:,i] = ux * cx[i] + uy * cy[i]
    uc2 = uc ** 2
    u2 = ux ** 2 + uy ** 2

    for i in range(q):
        feq[:,:,i] = rho * w_i[i] * (1 + uc / c_sqrd + (uc2 - c_sqrd * u2) / (2*c_sqrd **2))

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

    f_out[not_boundary,1] += 0.001

    for i in range(q):
        f_i[:, :, i] = np.roll(np.roll(f_out[:, :, i], cx[i], axis=0), cy[i], axis=1)


    rho = np.sum(f_i, axis=2)

    ux = (np.sum(f_i[:, :, [1, 5, 8]], axis=2) - np.sum(f_i[:, :, [3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:, :, [2, 5, 6]], axis=2) - np.sum(f_i[:, :, [4, 7, 8]], axis=2)) / rho


    u_sqrd = np.power(ux, 2) + np.power(uy, 2)
    uxplusuy = ux + uy
    minuxplusuy = -ux + uy
    minuxminuy = -uxplusuy
    plusuxminuy = -minuxplusuy

    minuxplusuy2c2 = np.power(minuxplusuy, 2)
    plusuxminuy2c2 = minuxplusuy2c2
    uxplusuy2c2 = np.power(uxplusuy, 2)
    minuxminuy2c2 = uxplusuy2c2
    ux2c2 = np.power(ux / c_sqrd, 2)
    uy2c2 = np.power(uy / c_sqrd, 2)

    # f_eq[:, :, 0] = w_i[0] * rho * (1 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 1] = w_i[1] * rho * (1 + ux / c_sqrd + 0.5 * ux2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 2] = w_i[2] * rho * (1 + uy / c_sqrd + 0.5 * uy2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 3] = w_i[3] * rho * (1 - ux / c_sqrd + 0.5 * ux2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 4] = w_i[4] * rho * (1 - uy / c_sqrd + 0.5 * uy2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 5] = w_i[5] * rho * (1 + uxplusuy / c_sqrd + 0.5 * uxplusuy2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 6] = w_i[6] * rho * (1 + minuxplusuy / c_sqrd + 0.5 * minuxplusuy2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 7] = w_i[7] * rho * (1 + minuxminuy / c_sqrd + 0.5 * minuxminuy2c2 - u_sqrd / (2 * c_sqrd))
    # f_eq[:, :, 8] = w_i[8] * rho * (1 + plusuxminuy / c_sqrd + 0.5 * plusuxminuy2c2 - u_sqrd / (2 * c_sqrd))

    f_eq = equilibrium(rho, ux, uy)





    if (t % 50 == 0):
        plt.clf();
        plt.imshow((ux ** 2 + uy ** 2).transpose(), cmap=cm.cool)
        plt.savefig("vel." + str(t / 100).zfill(4) + ".png")
