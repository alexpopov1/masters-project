
# Required packages
using Plots                               # For plotting results
using Distributed                         # For parallelising
using Statistics                          # For mean calculation
using LinearAlgebra                       # For norm calculation
@everywhere using JuMP                    # For optimisation
@everywhere using Ipopt                   # For optimisation


# Required files
include("Centralised.jl")
include("FormationModel.jl")



# ****************************************************************************************************************

# MODEL PROPERTIES
num = 8             # Number of agents
T = 50              # Fixed time horizon
N = 10 * T          # Number of time discretisations
rmin = 0.5          # Minimum distance between any two agents
v_final = [2, 0]    # Final velocity

umax = 0.15         # Maximum input
vmax = 5            # Maximum speed

xc, yc = 50, 10     # Centre coordinates of terminal circle
rc = 0.75           # Radius of terminal circle

init = [[0, 1, -1, 0.2],        # Initial conditions                      
        [7, 3, -1, 0.15],
        [-1, 5, -1, 0.1],
        [0, 7, -1, 0.05],
        [2, 12, -1, 0],
        [-3, 13, -1, -0.05],
        [1, 16, -1, -0.1],
        [6, 20, -1, -0.15]]


solve_method = 1

# ****************************************************************************************************************



parameters = num, T, N, init, v_final, (umax, vmax, rmin, xc, yc, rc)

states, inputs = Dict(), Dict()


if solve_method == 1
    states, inputs = remotecall_fetch(centralised, 2, parameters)
end




values = Dict()
agent_pairs = pairing(Array(1:num))
critical_pairs = []

for pair in agent_pairs
    values[pair] = coupled_inequalities(states, [pair], parameters)
    if maximum(values[pair]) > 0
        push!(critical_pairs, pair)
    end
end




function circle_plot(R, xc, yc)

        x = Array(range(-R, stop = R, length = 100)) 
        y_upper = (R^2 .- x.^2).^0.5
        y_lower = -(R^2 .- x.^2).^0.5

        plot(x.+xc, y_upper.+yc, linecolor=:blue, legend=false, aspect_ratio=:equal)
        plot!(x.+xc, y_lower.+yc, linecolor=:blue)

        return Nothing

end



circle_plot(rc, xc, yc)
trajectory_plot = plot!(states[1][1,:], states[1][2,:], color=:red, legend=false, grid=false, linewidth=0.5)
plot!([states[1][1,1]], [states[1][2,1]], seriestype=:scatter, markercolor=:red, markerstrokecolor=:red, markersize=3)
plot!([states[1][1,N+1]], [states[1][2,N+1]], seriestype=:scatter, markercolor=:red, markerstrokecolor=:red, markersize=3)


for i = 2:num
    plot!(states[i][1,:], states[i][2,:], color=:red, legend=false, linewidth=0.5)
    plot!([states[i][1,1]], [states[i][2,1]], seriestype=:scatter, markercolor=:red, markerstrokecolor=:red, markersize=3)
    plot!([states[i][1,N+1]], [states[i][2,N+1]], seriestype=:scatter, markercolor=:red, markerstrokecolor=:red, markersize=3)
end

display(trajectory_plot)











