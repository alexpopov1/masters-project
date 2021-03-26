

# Required packages
using Plots                               # For plotting results
using Distributed                         # For parallelising 
@everywhere using JuMP                    # For optimisation
@everywhere using Ipopt                   # For optimisation


# Required files
include("SmallestNeighbourhood.jl")
include("Centralised.jl")







@everywhere function make_neighbourhood(num_cars::Int, num_followers::Array{Int64, 1}, num_leaders::Array{Int64, 1})
    
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
num_cars = 8                                         # Number of cars
dstart = Array(range(8, stop = 8, length = num_cars))  # Relative starting positions
dsep = 2                                               # Initial and final distance between consecutive cars
D = 500                                                # Total distance travelled by each car
radius = 16                                            # Radius of ring
dmin = 0.5                                             # Minimum distance between consecutive cars
umax = 0.1                                             # Maximum force
umin = -0.2                                            # Minimum force
vmax = 200                                             # Maximum velocity
vmin = -vmax                                           # Minimum velocity
T = 100                                              # Fixed time horizon
N = 10 * T                                             # Number of time discretisations
v1 = [0,2,0,-2,0,0,0,0]                           # Initial velocities
v2 = Array(range(0, stop=0, length=num_cars))          # Terminal velocities


# GRAPH PROPERTIES
num_followers = repeat([1], num_cars)                  # Number of leaders
num_leaders = repeat([1], num_cars)                    # Number of followers

# num_followers = [1, 1, 2, 3, 1, 1, 1, 1]
# num_leaders = [3, 2, 1, 0, 1, 1, 1, 1]
hub = Int(ceil(num_cars/2))                            # Hub agent


# SOLVER PROPERTIES (see KEY)
iter_limit = 100                                    # Ipopt iteration limit
solve_method = 2                                      # Solve method flag


# KEY
# 1: centralised 
# 2: smallest neighbour 



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




states, inputs, testing = Dict(), Dict(), Dict()      # Initialise solutions






if solve_method == 1

    @time states, inputs = centralised(parameters)

elseif solve_method == 2


    agent_procs = Dict(i=>workers()[i] for i = 1:parameters[1])
    @time @sync for sys = 1:num_cars

        @async states[sys], inputs[sys], testing[sys] = remotecall_fetch(smallest_neighbourhood, agent_procs[sys],
                                                   sys, hub, parameters, neighbours[sys], iter_limit = iter_limit)

    end

end





t = range(0,stop=T,length=N+1)
trajectories = [states[i][1, :] for i = 1:num_cars]
inputs = [inputs[i] for i = 1:num_cars]



trajectoryPlot = plot(t, trajectories, legend=false)
display(trajectoryPlot)



# Find input cost
inputCost = sum(sum([inputs[i] .* inputs[i] for i = 1:num_cars]))
println("Input cost = ", inputCost)





















