import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

# General flow constants
nx = 400 # number of cells in x-direction
ny = 100 # number of cells in y-direction
uMax = 0.1 # maximum velocity
Re = 10 # Reynolds number
maxIter = 1000 # Total number of iterations
rho = 1 # density
obstacle_x = nx/2 # x locaction of cylinder
obstacle_y = ny/2 # y location of cylinder
obstacle_r = ny/9 # radius of the cylinder
nu = uMax*2.*obstacle_r/Re # kinematic viscosity
tau = 1./(3*nu+0.5) # relaxation parameter
q = 9 # lattice population

# D2Q9 Lattice
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
e_i = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [1,-1], [-1,-1], [1,-1]])
opposites = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
pos_inlet = 1
pos_outlet = nx
col = range(1,(ny-1))


y, x = np.meshgrid(range(ny),range(nx))

# Set boundaries
def set_boundary(max_x, max_y, radius2):
    b = np.zeros((max_x,max_y))
    for i in range(max_x):
        for j in range(max_y):
            b[i,j] = (i-obstacle_x)**2+(j-obstacle_y)**2 <= radius2
    b[:, 0] = True
    b[:, -1] = True
    return b == 1

boundary = set_boundary(nx,ny,obstacle_r**2)
not_boundary = boundary == False

# Initial conditions
L = ny-2
ux = 4*uMax/(L*L) * (y*L-y*y) # impose Poiseuille flow
uy = np.zeros((nx,ny))

def set_initial_conditions(max_x,max_y):
    uc = np.zeros((max_x,max_y,q))
    f_i = np.zeros((max_x, max_y, q))
    c_sqrd = 1/3
    for i in range(q):
        uc[:,:,i] = cx[i]*ux+cy[i]*uy
        f_i[:,:,i] = rho*w_i[i]*(1+uc[:,:,i]/c_sqrd+(uc[:,:,i]*uc[:,:,i])/(2*c_sqrd)-(ux*ux+uy*uy)/(2*c_sqrd))
    return f_i

# Main loop
def inlet(max_y):
    for y in range(max_y):
        ux[y] = (4*uMax / (L*L)) * (1 - (np.abs(y-ny/2) / (ny / 2)) ** 2)
    return ux[0,:]

f_i = set_initial_conditions(nx, ny)
f_eq = np.copy(f_i)
f_out = np.copy(f_i)
ux[0,:] = inlet(ny)
ux[boundary] = 0
uy[boundary] = 0

for cycle in range(maxIter):
    for i in range(q):
        f_i[:, :, i] = np.roll(np.roll(f_out[:, :, i], cx[i], axis=0), cy[i], axis=1)
    f_i[0] = f_i[1] # Taking away periodic boundary conditions on the inflow
    f_i[-1] = f_i[-2] # And also at the outflow

    for i in range(q):
        f_i[boundary, i] = f_i[boundary, opposites[i]]

    for i in range(nx):
        ux[i,not_boundary[i,:]] += 1.0 * ux[0, not_boundary[i,:]]

    # Macroscopic variables
    rho = np.sum(f_i, axis=2)
    ux = (np.sum(f_i[:,:,[1, 5, 8]], axis=2) - np.sum(f_i[:,:,[3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:,:,[2, 5, 6]], axis=2) - np.sum(f_i[:,:,[4, 7, 8]], axis=2)) / rho

    if (cycle % 50 == 0):
        plt.clf();
        plt.imshow((ux).transpose(), cmap=cm.Blues)
        # plt.clim(0,uLB)
        plt.colorbar()
        plt.savefig("velx/" + str(cycle / 100).zfill(4) + ".png")
        plt.clf();
        plt.imshow((uy).transpose(), cmap=cm.Blues)
        plt.colorbar()
        plt.savefig("vely/" + str(cycle / 100).zfill(4) + ".png")
        plt.clf();
        plt.imshow((np.sqrt(ux ** 2 + uy ** 2)).transpose(), cmap=cm.Blues)
        # plt.clim(0,uLB)
        plt.colorbar()
        plt.savefig("vel/" + str(cycle / 100).zfill(4) + ".png")


print(ux.shape)
print(tau)



