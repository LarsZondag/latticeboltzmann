include("steps.jl")

# Define the velocity vector:
diag = 1/sqrt(2)
e_i = [0 0; 1 0; 0 1; -1 0; 0 -1; diag diag; -diag diag; -diag -diag; diag -diag]

# Define the weights:
w = [4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36]

ρ, u, f_i, f_i_eq = initialize_f_lattice(2, 2)
