import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

maxIter = 40 # Total number of time iterations.
Re      = 220.0  # Reynolds number.
nx = 50
ny = 20
q = 9 # Lattice dimensions and populations.
r=ny/9  # Coordinates of the cylinder.
uLB     = 0.04 # Velocity in lattice units.
nulb    = uLB*r/Re
tau = 1.0 / (3.*nulb+0.5) # Relaxation parameter.

e_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

inlet_velo = 10**-7

c_sqrd = 1/3

rho = 1;

f_i = np.zeros((nx, ny, q))

for i in range(nx):
    for j in range(ny):
        f_i[i, j, :] = w_i * rho

f_eq = np.copy(f_i)

def calc_f_eq(velocity, density, velo_sqrd):
    s_i = np.zeros((nx, ny, q))
    for i in range(nx):
        for j in range(ny):
            for k in range(q):
                e_i_velo = np.dot(e_i[k], velocity[i, j, :])
                s_i[i, j, :] += 3 * e_i_velo / 2 / c_sqrd  + 9 / 2 * e_i_velo ** 2 / c_sqrd



for t in range(maxIter):

    for i in range(q): # Streaming step.
        f_i[i,:,:] = np.roll(np.roll(fout[i,:,:],e_i[i,0],axis=0),e_i[i,1],axis=1)


    rho = np.sum(f_i, axis=2)

    ux = np.divide((np.sum(f_i[:,:,[1, 5, 8]], axis=2) - np.sum(f_i[:,:,[3, 6, 7]], axis=2)), rho)
    uy = np.divide((np.sum(f_i[:,:,[2, 5, 6]], axis=2) - np.sum(f_i[:,:,[4, 7, 8]], axis=2)), rho)
    ux[0,:] =+ inlet_velo
    u_sqrd = np.power(ux, 2) + np.power(uy, 2)








