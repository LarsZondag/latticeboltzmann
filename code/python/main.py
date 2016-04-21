import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

maxIter = 1000 # Total number of time iterations.
Re      = 0.1  # Reynolds number.
nx = 500
ny = 200
q = 9 # Lattice dimensions and populations.
r=ny/9  # Coordinates of the cylinder.
uLB     = 0.1 # Velocity in lattice units.
nulb    = uLB*r/Re
tau = 1.0 / (3.*nulb+0.5) # Relaxation parameter.

e_i = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
w_i = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
to_right = np.array([1, 5, 8])
to_left = np.array([3, 6, 7])


c_sqrd = 1/3

rho = 1;

f_i = np.zeros((nx, ny, q))

for i in range(nx):
    for j in range(ny):
        f_i[i, j, :] = w_i * rho

f_eq = np.copy(f_i)

def equilibrium(rho,u):              # Equilibrium distribution function.
    usqr = 3./2.*(u[:,:,0]**2+u[1]**2)
    feq = np.zeros((nx,ny,q))
    cu = np.zeros((nx, ny, q))
    for i in range(nx):
        for j in range(ny):
            cu[i, j, :] += 3.0 * np.dot(e_i, u) * w_i
    for i in range(q):
        feq[:,:,i] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq

def set_boundary(max_x, max_y, radius2):
    center = [max_x/2, max_y/2]
    b = np.zeros((max_x, max_y))
    for i in range(max_x):
        for j in range(max_y):
            b[i, j] = (i-center[0]) ** 2 + (j-center[1]) ** 2 <= radius2
    b[:, 0] = True
    b[:, -1] = True
    return b == 1

boundary = set_boundary(nx, ny, (ny/6) ** 2)

BB = np.zeros((nx, ny, q))

for t in range(maxIter):

    rho = np.sum(f_i, axis=2)

    ux = (np.sum(f_i[:,:,[1, 5, 8]], axis=2) - np.sum(f_i[:,:,[3, 6, 7]], axis=2)) / rho
    uy = (np.sum(f_i[:,:,[2, 5, 6]], axis=2) - np.sum(f_i[:,:,[4, 7, 8]], axis=2)) / rho

    ux[boundary] = 0
    uy[boundary] = 0
    ux[0,1:-1] = uLB

    u_sqrd = np.power(ux, 2) + np.power(uy, 2)
    uxplusuy = ux+uy
    minuxplusuy = -ux+uy
    minuxminuy = -uxplusuy
    plusuxminuy = -minuxplusuy

    minuxplusuy2c2 = np.power(minuxplusuy, 2)
    plusuxminuy2c2 = minuxplusuy2c2
    uxplusuy2c2 = np.power(uxplusuy, 2)
    minuxminuy2c2 = uxplusuy2c2
    ux2c2 = np.power(ux/c_sqrd,2)
    uy2c2 = np.power(uy/c_sqrd,2)

    f_eq[:,:,0]=w_i[0]*rho * (1 -u_sqrd/(2*c_sqrd))
    f_eq[:,:,1]=w_i[1]*rho * (1+ux/c_sqrd+0.5*ux2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,2]=w_i[2]*rho * (1+uy/c_sqrd+0.5*uy2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,3]=w_i[3]*rho * (1-ux/c_sqrd+0.5*ux2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,4]=w_i[4]*rho * (1-uy/c_sqrd+0.5*uy2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,5]=w_i[5]*rho * (1+uxplusuy/c_sqrd+0.5*uxplusuy2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,6]=w_i[6]*rho * (1+minuxplusuy/c_sqrd+0.5*minuxplusuy2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,7]=w_i[7]*rho * (1+minuxminuy/c_sqrd+0.5*minuxminuy2c2-u_sqrd/(2*c_sqrd))
    f_eq[:,:,8]=w_i[8]*rho * (1+plusuxminuy/c_sqrd+0.5*plusuxminuy2c2-u_sqrd/(2*c_sqrd))


    f_out = f_i - (f_i - f_eq)/tau

    for i in range(q):
        f_i[boundary, i] = f_i[boundary, opp[i]]


    for i in range(q):
        f_i[:,:,i] = np.roll(np.roll(f_out[:,:,i],e_i[i,0],axis=0),e_i[i,1],axis=1)

    if (t%50==0): # Visualization
        plt.clf(); plt.imshow((ux**2+uy**2).transpose(),cmap=cm.cool)
        plt.savefig("vel."+str(t/100).zfill(4)+".png")

