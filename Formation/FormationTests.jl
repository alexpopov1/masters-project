
# Required packages
using Plots                               # For plotting results
using Distributed                         # For parallelising
using Statistics                          # For mean calculation
using LinearAlgebra                       # For norm calculation
@everywhere using JuMP                    # For optimisation
@everywhere using Ipopt                   # For optimisation


# Required files
include("FormationModel.jl")
include("Centralised.jl")
include("SmallestNeighbourhood.jl")
include("Consensus.jl")


@everywhere function make_graph(num::Int, init::Array, num_of_neighbours::Int)

    neighbours = Dict()

    for i = 1:num

        distance = Dict()
        for j in filter(x->x!=i, Array(1:num))
            distance[j] = sqrt((init[i][1]-init[j][1])^2 + (init[i][2]-init[j][2])^2)
        end

        rank = [sort(collect(distance), by=x->x[2])[j][1] for j=1:num-1]
        # neighbours[i] = rank[1:num_of_neighbours]
        neighbours[i] = [j for j in filter(x->x!=i, Array(1:num)) if distance[j] <= 8]

    end

    return neighbours

end
        
            
             


# ****************************************************************************************************************

# MODEL PROPERTIES
num = 8             # Number of agents
hub = 4
T = 50              # Fixed time horizon
N = 10 * T          # Number of time discretisations
rmin = 0.5          # Minimum distance between any two agents
v_final = [2, 0]    # Final velocity

umax = 0.15         # Maximum input
vmax = 5            # Maximum speed

xc, yc = 50, 10     # Centre coordinates of terminal circle
rc = 0.75           # Radius of terminal circle

init = [[0, 1, 1, 0.2],        # Initial conditions                      
        [7, 3, 1, 0.15],
        [-1, 5, 1, 0.1],
        [0, 7, 1, 0.05],
        [2, 12, 1, 0],
        [-3, 13, 1, -0.05],
        [1, 16, 1, -0.1],
        [6, 20, 1, -0.15]]

num_of_neighbours = 3


iter_limit = 1000


solve_method = 3

# ****************************************************************************************************************



neighbours = make_graph(num, init, num_of_neighbours)
parameters = num, T, N, init, v_final, (umax, vmax, rmin, xc, yc, rc)
states, inputs, timing = Dict(), Dict(), Dict()
agent_procs = Dict(i=>workers()[i] for i = 1:parameters[1])

testing = Dict()



if solve_method == 1

    states, inputs = remotecall_fetch(centralised, 2, parameters)




elseif solve_method == 2
    
    @sync for sys = 1:num
        @async states[sys], inputs[sys], timing[sys] = remotecall_fetch(smallest_neighbourhood, agent_procs[sys],
                                                   sys, hub, parameters, neighbours[sys], iter_limit = iter_limit)
    end

    println("TOTAL TIME TO COMPLETION: ", mean([timing[j] for j = 1:num]))




elseif solve_method == 3

    @sync for sys = 1:num
        @async states[sys], inputs[sys], testing[sys] = remotecall_fetch(consensus, agent_procs[sys],
                                                                         sys, parameters, neighbours[sys])
    end


    history = Dict()
    for i = 1:num
        history[i] = testing[i]
    end

    iterations = 2
    violation_norm = Array{Float64, 1}(undef, iterations)

    for k = 1:iterations
        tracking = Dict(i=>history[i][k] for i = 1:num)
        con_values = coupled_inequalities(tracking, pairing(Array(1:num)), parameters)
        indices = findall(val->val>0, con_values)
        violations = [con_values[i] for i in indices]
        violation_norm[k] = norm(violations)
    end


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











