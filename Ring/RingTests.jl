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
@everywhere include("RingModel.jl")
@everywhere include("WarmStart.jl")



function make_neighbourhood(num_cars::Int, num_followers::Array{Int64, 1}, num_leaders::Array{Int64, 1})

    leaders = Array{Array{Int64}}(undef, num_cars)
    followers = Array{Array{Int64}}(undef, num_cars)
    neighbours = Array{Array}(undef, num_cars)

    for i = 1:num_cars

        # Set followers for system i
        if num_followers[i] == 0
	    followers[i] = []
        elseif i <= num_followers[i]
            followers[i] = [([Array((num_cars-(num_followers[i]-i)):num_cars), Array(1:i-1)]...)...]
        else
            followers[i] = Array(i-num_followers[i]:i-1)
        end

        # Set leaders (predecessors) for system i
        if num_leaders[i] == 0
            leaders[i] = []
        elseif i > num_cars - num_leaders[i]
            leaders[i] = [([Array(i+1:num_cars), Array(1:(num_leaders[i]-(num_cars-i)))]...)...]
        else
            leaders[i] = Array(i+1:i+num_leaders[i])
        end

        # Define total neighbourhood for system i
        neighbours[i] =[([followers[i], leaders[i]]...)...]

    end

    return neighbours

end














# ****************************************************************************************************************

# MODEL PROPERTIES
num_cars = 32                                          # Number of cars
dstart = Array(range(8, stop = 8, length = num_cars))  # Relative starting positions
dsep = 2                                               # Initial and final distance between consecutive cars
D = 500                                                # Total distance travelled by each car
radius = 200                                           # Radius of ring
dmin = 0.5                                             # Minimum distance between consecutive cars
umax = 0.1                                             # Maximum force
umin = -0.2                                            # Minimum force
vmax = 200                                             # Maximum velocity
vmin = -vmax                                           # Minimum velocity
T = 100                                                # Fixed time horizon
N = 10 * T                                             # Number of time discretisations
# v1 = v2 = Array(range(0, stop=0, length=num_cars))   # Initial velocities
# v1 = [([[1,0,-1], repeat([0],num_cars-3)]...)...]
v1 = [([reverse(Array(range(1, length=Int(num_cars/2), step=2))), 
        Array(range(-1, length=Int(num_cars/2), step=-2))]...)...]

v2 = Array(range(0, stop=0, length=num_cars))          # Terminal velocities


# GRAPH PROPERTIES
num_followers = repeat([1], num_cars)                  # Number of leaders
num_leaders = repeat([1], num_cars)                    # Number of followers
num_followers[1] = 0
num_leaders[num_cars] = 0
# num_followers = [1, 1, 2, 3, 1, 1, 1, 1]
# num_leaders = [3, 2, 1, 0, 1, 1, 1, 1]
hub = Int(ceil(num_cars/2))                            # Hub agent


# SOLVER PROPERTIES (see KEY)
iter_limit = 100                                       # Ipopt iteration limit
solve_method = 5                                    # Solve method flag


# KEY
# 1: centralised
# 2: smallest neighbour
# 3: consensus
# 4: ADMM
# 5: algorithm

# ****************************************************************************************************************












omega_min = vmin / radius
omega_max = vmax / radius
omega1 = v1 ./ radius
omega2 = v2 ./ radius
astart = dstart ./ radius
amin = dmin / radius
asep = dsep / radius
atot = D / radius

parameters = num_cars, T, N, (umin, umax), (omega_min, omega_max), (omega1, omega2), astart, amin, asep, atot
neighbours = make_neighbourhood(num_cars, num_followers, num_leaders)
agent_procs = Dict(i=>workers()[i] for i = 1:parameters[1])

states, inputs,  = Dict(), Dict()   
timing, testing, history = Dict(), Dict(), Dict() 





if solve_method == 1

    @time states, inputs = remotecall_fetch(centralised, 2, parameters)





elseif solve_method == 2

    @time @sync for sys = 1:num_cars
        @async states[sys], inputs[sys], timing[sys] = remotecall_fetch(smallest_neighbourhood, agent_procs[sys],
                                                   sys, hub, parameters, neighbours[sys], iter_limit = iter_limit)
    end

    println("TOTAL TIME TO COMPLETION: ", mean([timing[j] for j = 1:num_cars]))





elseif solve_method == 3

    historyset = Dict()
    @sync for sys = 1:num_cars
        @async states[sys], inputs[sys], historyset[sys], timing[sys] = remotecall_fetch(consensus, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys],
                                                                         max_iterations = 5)
    end
    
    history = Dict(j=>Dict(i=>historyset[j][i][1] for i=1:length(historyset[1])) for j = 1:num_cars)
    z_track = Dict(j=>Dict(i=>historyset[j][i][2] for i=1:length(historyset[1])) for j = 1:num_cars)


elseif solve_method == 4

    @sync for sys = 1:num_cars
        @async states[sys], inputs[sys], history[sys], timing[sys] = remotecall_fetch(ADMM, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys],
                                                                         rho = 1.0, max_iterations = 50)
    end


elseif solve_method == 5

    @sync for sys = 1:num_cars
        @async states[sys], inputs[sys], history[sys], timing[sys], testing[sys] = remotecall_fetch(algorithm, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys],
                                                                         max_iterations = 50)
    end

end



if solve_method in [3, 4, 5]

    println("Total time: ", maximum([timing[i] for i = 1:num_cars]))
    iterations = length(history[1])
    violation_norm = Array{Float64, 1}(undef, iterations)

    for k = 1:iterations

        tracking = [history[i][k] for i = 1:num_cars]
        violation_vector = []

        for j = 1:num_cars

            islap = j == num_cars ? true : false
            con_values = coupled_inequalities(tracking[j], tracking[j%num_cars+1], parameters, islap)
            indices = findall(i->i>0, con_values)            
            violations = isempty(indices) ? [0] : [con_values[i] for i in indices]
            violation_vector = vcat(violation_vector, violations)

        end

        violation_norm[k] = norm(violation_vector)

    end

end






t = range(0,stop=T,length=N+1)
trajectories = [states[i][1, :] for i = 1:num_cars]
inputsworkers = [inputs[i] for i = 1:num_cars]

trajectoryPlot = plot(t, trajectories, legend=false)
display(trajectoryPlot)

scaled_error = violation_norm / violation_norm[1]
resplot = plot(1:iterations, scaled_error, xlab="Iterations", ylab="Scaled error", legend=false)


# Find input cost
inputCost = sum(sum([inputs[i] .* inputs[i] for i = 1:num_cars]))
println("Input cost = ", inputCost)
