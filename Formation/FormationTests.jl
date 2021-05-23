


# Required packages
using Plots                               # For plotting results
using Distributed                         # For parallelising
using Statistics                          # For mean calculation
using LinearAlgebra                       # For norm calculation
@everywhere using JuMP                    # For optimisation
@everywhere using Ipopt                   # For optimisation


# Required files
@everywhere include("SmallestNeighbourhood.jl")
@everywhere include("Centralised.jl")
@everywhere include("ADMM.jl")
@everywhere include("Consensus.jl")
@everywhere include("Algorithm.jl")
@everywhere include("FormationModel.jl")
@everywhere include("WarmStart.jl")
@everywhere include("SpineSweep.jl")


@everywhere function make_graph(num::Int, init::Array, num_of_neighbours::Int)

    neighbours = Dict()

    for i = 1:num

        distance = Dict()
        for j in filter(x->x!=i, Array(1:num))
            distance[j] = sqrt((init[i][1]-init[j][1])^2 + (init[i][2]-init[j][2])^2)
        end

        # rank = [sort(collect(distance), by=x->x[2])[j][1] for j=1:num-1]
        # neighbours[i] = rank[1:num_of_neighbours]
        neighbours[i] = [j for j in filter(x->x!=i, Array(1:num)) if distance[j] <= 2.5]

    end

    return neighbours

end
        
            
             


# ****************************************************************************************************************

# MODEL PROPERTIES
num = 8             # Number of agents
hub = 4
T = 50              # Fixed time horizon
N = 10 * T          # Number of time discretisations
rmin = 0.2          # Minimum distance between any two agents
v_final = [2, 0]    # Final velocity

umax = 0.15         # Maximum input
vmax = 5            # Maximum speed

xc, yc = 50, 15     # Centre coordinates of terminal circle
rc = 1          # Radius of terminal circle

init = [[0, 1, 1, 0.2],        # Initial conditions                      
        [1, 2, 1, 0.15],
        [-1, 3, 1, 0.1],
        [0, 4, 1, 0.05],
        [1, 5, 1, 0],
        [-1, 6, 1, -0.05],
        [0, 7, 1, -0.1],
        [1, 8, 1, -0.15]]

num_of_neighbours = 2
path = [2, 4, 5, 7]

iter_limit = 1000
solve_method = 1


# KEY
# 1: centralised
# 2: smallest neighbour
# 3: consensus
# 4: ADMM
# 5: algorithm
# 6: spinesweep

# ****************************************************************************************************************







neighbours = make_graph(num, init, num_of_neighbours)
parameters = num, T, N, init, v_final, (umax, vmax, rmin, xc, yc, rc)
agent_procs = Dict(i=>workers()[i] for i = 1:parameters[1])

states, inputs = Dict(), Dict()
timing, testing, history = Dict(), Dict(), Dict()


if solve_method == 1

    states, inputs, timing = remotecall_fetch(centralised, 2, parameters)
    println("Total time: ", timing)



elseif solve_method == 2
    
    @sync for sys = 1:num
        @async states[sys], inputs[sys], timing[sys] = remotecall_fetch(smallest_neighbourhood, agent_procs[sys],
                                                   sys, hub, parameters, neighbours[sys], iter_limit = iter_limit)
    end

    println("TOTAL TIME TO COMPLETION: ", mean([timing[j] for j = 1:num]))



elseif solve_method == 3


    @sync for sys = 1:num
        @async states[sys], inputs[sys], history[sys], timing[sys] = remotecall_fetch(consensus, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys])
    end




elseif solve_method == 4

    @sync for sys = 1:num
        @async states[sys], inputs[sys], history[sys], timing[sys] = remotecall_fetch(ADMM, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys])
    end



elseif solve_method == 5

    @sync for sys = 1:num
        @async states[sys], inputs[sys], history[sys], timing[sys], testing[sys] = remotecall_fetch(algorithm, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys],
                                                                         max_iterations = 50)
    end




elseif solve_method == 6

    @sync for sys = 1:num
        @async states[sys], inputs[sys], timing[sys] = remotecall_fetch(spinesweep, agent_procs[sys], 
                                                           sys, parameters, neighbours[sys], path)
    end
    println("DONE!")
    println("Total time: ", maximum([timing[i] for i = 1:num]))
    println("Max violation = ", maximum(coupled_inequalities(states, pairing(Array(1:num)), parameters)))

end



if solve_method in [3, 4, 5]
   
    println("Total time: ", maximum([timing[i] for i = 1:num]))
    iterations = length(history[1])
    violation_norm = Array{Float64, 1}(undef, iterations)
    prim_res_norm = Array{Float64, 1}(undef, iterations)

    for k = 1:iterations

        x_track = Dict(i=>history[i][k][1] for i = 1:num)
        z_track = Dict(i=>history[i][k][2] for i = 1:num)
        con_values = coupled_inequalities(x_track, pairing(Array(1:num)), parameters)
        indices = findall(val->val>0, con_values)
        violations = [con_values[i] for i in indices]
        violation_norm[k] = norm(violations)
        prim_res_norm[k] = norm(vcat([(x_track[i] - z_track[i]) for i = 1:num]...))

    end


    scaled_error = violation_norm / violation_norm[1]
    scaled_residual = prim_res_norm / prim_res_norm[1]
    viplot = plot(1:iterations, scaled_error, xlab="Iterations", ylab="Scaled error", legend=false)
    resplot = plot(1:iterations, scaled_residual, xlab="Iterations", ylab="Scaled consensus residual", legend=false)

end






function circle_plot(R, xc, yc)

        x = Array(range(-R, stop = R, length = 100)) 
        y_upper = (R^2 .- x.^2).^0.5
        y_lower = -(R^2 .- x.^2).^0.5

        plot(x.+xc, y_upper.+yc, linecolor=:blue, legend=false, 
             aspect_ratio=:equal, linestyle=:dot)
        plot!(x.+xc, y_lower.+yc, linecolor=:blue, linestyle=:dot)

        return Nothing

end



circle_plot(rc, xc, yc)

trajectory_plot = plot!(states[1][1,:], states[1][2,:], 
                  color=:grey, legend=false, grid=false, linewidth=0.5,
                  ylims=(0,20))
plot!([states[1][1,1]], [states[1][2,1]], seriestype=:scatter,
      markercolor=:red, markerstrokecolor=:red, markersize=2)
plot!([states[1][1,N+1]], [states[1][2,N+1]], seriestype=:scatter, 
     markercolor=:red, markerstrokecolor=:red, markersize=2)


for i = 2:num
    plot!(states[i][1,:], states[i][2,:], color=:grey, 
          legend=false, linewidth=0.5)
    plot!([states[i][1,1]], [states[i][2,1]], seriestype=:scatter, 
          markercolor=:red, markerstrokecolor=:red, markersize=2)
    plot!([states[i][1,N+1]], [states[i][2,N+1]], seriestype=:scatter, 
          markercolor=:red, markerstrokecolor=:red, markersize=2)
end

display(trajectory_plot)


# Find input cost
inputCost = sum(sum([inputs[i] .* inputs[i] for i = 1:num]))
println("Input cost = ", inputCost)

