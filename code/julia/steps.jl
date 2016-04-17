# export initialize_f_lattice, streaming, compute_macroscopic
# export compute_f_equilibrium, collision
include("static.jl")

function initialize_f_lattice(x_length, y_length)
  f_i = zeros(x_length, y_length, 9)
  f_i_eq = zeros(x_length, y_length, 9)
  ρ = zeros(x_length, y_length)
  u = zeros(x_length, y_length, 3)
  return ρ, u, f_i, f_i_eq
end

function streaming()

end

function compute_macroscopic()

end

function compute_f_equilibrium()

end

function collision()

end
