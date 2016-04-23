# D2Q9 Lattice Boltzmann

The script can be found at `code/python/main.py`.

In this script a D2Q9 Lattice Boltzmann simulation is run. The standard gemeomtry is that of a pipe. The parameter `cylinder` can be switched from `True` to `False` to place a cylinderical object in the flow or not. The results of the flow will be saved in the folders `vel`, `velx`, `vely`, and `profile` for the norm of the velocity, the x-velocity, the y-velocity, and the vertical velocity profile at the middle of the grid.

# Results 
## The development into Poiseuille flow: 

Here a constant pressure drop was simulated by increasing the horizontal velocity at each lattice-node at each time step. No object was placed in the flow.
https://youtu.be/Fg2Jket9hvY

## Flow around a cylinder:
Here the same pressure drop was applied. Cylindrical object was placed with bounce-back boundary conditions. The flow speed is increased until flow becomes turbulent. 
https://youtu.be/TkTssb7E_0Y
