import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

lx = 200
ly = 100
obst_x = lx/2
obst_y = ly/2
obst_r = ly/10
uMax = 0.1
Re = 100
nu = uMax * 2 * obst_r / Re
omega = 1 / (3 * nu + 1/2)
maxT = 4000
tPlot = 50
q = 9
c_sqrd = 3

# Lattice constants
t = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1/36, 1/36, 1/36, 1/36])
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
opp = np.array([0, 3, 4, 1, 3, 7, 8, 5, 6])
col = np.arange(1, ly-1)
x_inlet = 0
x_outlet = lx-1

xx = np.arange(lx)
yy = np.arange(ly)
x, y = np.meshgrid(xx, yy, sparse=False)
obst = (x - obst_x) ** 2 + (y - obst_y) ** 2 <= obst_r ** 2
obst[:,[0, ly-1]] = True
print('transposed',np.transpose(np.shape(obst)))

# Initial conditions
L = ly-2
y_phys = y-1.5
ux = 4 * uMax / L ** 2 * (y_phys * L - y_phys ** 2)
uy = np.zeros((lx, ly))
print(np.shape(ux))
print(np.shape(uy))
rho = 1
fIn = np.zeros((lx, ly, q))

u_sqrd = np.power(ux, 2) + np.power(uy, 2)
uxplusuy = ux+uy
minuxplusuy = -ux+uy
minuxminuy = -uxplusuy
plusuxminuy = -minuxplusuy

fIn[:,:,0]=t[0]*rho * (1 -u_sqrd/(2*c_sqrd))
fIn[:,:,1]=t[1]*rho * (1+ux/c_sqrd+0.5*np.power(ux/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,2]=t[2]*rho * (1+uy/c_sqrd+0.5*np.power(uy/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,3]=t[3]*rho * (1-ux/c_sqrd+0.5*np.power(ux/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,4]=t[4]*rho * (1-uy/c_sqrd+0.5*np.power(uy/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,5]=t[5]*rho * (1+uxplusuy/c_sqrd+0.5*np.power(uxplusuy/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,6]=t[6]*rho * (1+minuxplusuy/c_sqrd+0.5*np.power(minuxplusuy/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,7]=t[7]*rho * (1+minuxminuy/c_sqrd+0.5*np.power(minuxminuy/c_sqrd,2)-u_sqrd/(2*c_sqrd))
fIn[:,:,8]=t[8]*rho * (1+plusuxminuy/c_sqrd+0.5*np.power(plusuxminuy/c_sqrd,2)-u_sqrd/(2*c_sqrd))


for cycle in range(maxT):

    rho = np.sum(fIn, axis=2)
    ux = np.divide((np.sum(fIn[:,:,[1, 5, 8]], axis=2) - np.sum(fIn[:,:,[3, 6, 7]], axis=2)), rho)
    uy = np.divide((np.sum(fIn[:,:,[2, 5, 6]], axis=2) - np.sum(fIn[:,:,[4, 7, 8]], axis=2)), rho)

    y_phys = col - 1.5
    ux[x_inlet, col] = 4 * uMax / L ** 2 * (y_phys * L - y_phys ** 2)
    uy[x_inlet, col] = 0
    rho[x_inlet, 1:-2] = np.power((1 - ux[0, 1:-2]), -1) * (np.sum(fIn[0, 1:-2, [0, 2, 4]], axis=0) + 2 * np.sum(fIn[0, 1:-2, [3, 6, 7]], axis=0))

plt.contourf(x, y, ux ** 2 + uy ** 2)
plt.show()