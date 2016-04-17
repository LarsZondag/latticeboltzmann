function calculate_ρ(f_i)
 return sum(f_i)
end

function calculate_u(ρ, f_i, e_i)

end

function calculate_s(w_i, e_i, u, c)
  return w_i * (3 * dot(e_i, u) / c + 9 / 2 * (dot(e_i, u))^2 / c^2 - 3 / 2 * dot(u, u) / c^2)
end
