# Required packages
using Plots                               # For plotting results
using Distributed                         # For parallelising
using Statistics                          # For mean calculation
using LinearAlgebra                       # For norm calculation
@everywhere using JuMP                    # For optimisation
@everywhere using Ipopt                   # For optimisation


# Required files
@everywhere include("Centralised.jl")
@everywhere include("ADMM.jl")
@everywhere include("CDSBS.jl")
@everywhere include("PlatoonModel.jl")
@everywhere include("WarmStart.jl")


# Define neighbourhood for each agent
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

# MODEL PROPERTIES (platoon layout calculated from 'straightened' ring of agents)
num_cars = 32                                                       # Number of cars
dstart = Array(range(8, stop = 8, length = num_cars))               # Relative starting positions
dsep = 2                                                            # Initial and final distance between consecutive cars
D = 500                                                             # Total distance travelled by each car
radius = 200                                                        # Radius of ring
dmin = 0.5                                                          # Minimum distance between consecutive cars
umax = 0.1                                                          # Maximum force
umin = -0.2                                                         # Minimum force
vmax = 200                                                          # Maximum velocity
vmin = -vmax                                                        # Minimum velocity
T = 100                                                             # Fixed time horizon
N = 10 * T                                                          # Number of time discretisations  
v1 = [([reverse(Array(range(1, length=Int(num_cars/2), step=2))), 
        Array(range(-1, length=Int(num_cars/2), step=-2))]...)...]  # Initial velocities
v2 = Array(range(0, stop=0, length=num_cars))                       # Terminal velocities


# GRAPH PROPERTIES
num_followers = repeat([1], num_cars)                  # Number of leaders
num_leaders = repeat([1], num_cars)                    # Number of followers
num_followers[1] = 0
num_leaders[num_cars] = 0
hub = Int(ceil(num_cars/2))                            # Hub agent


# SOLVER PROPERTIES (see KEY)
iter_limit = 100                                       # Ipopt iteration limit
solve_method = 3                                       # Solve method flag


# KEY
# 1: centralised
# 2: ADMM
# 3: CDS-BS

# ****************************************************************************************************************

# Calculate model parameters for circle configuration
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

    @sync for sys = 1:num_cars
        @async states[sys], inputs[sys], history[sys], timing[sys] = remotecall_fetch(ADMM, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys],
                                                                         rho = 1.0, max_iterations = 50)
    end

			
elseif solve_method == 3

    @sync for sys = 1:num_cars
        @async states[sys], inputs[sys], history[sys], timing[sys], testing[sys] = remotecall_fetch(CDSBS, agent_procs[sys],
                                                                         sys, hub, parameters, neighbours[sys],
                                                                         max_iterations = 50)
    end

end



if solve_method in [2, 3]

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

# Prepare plotting variables
t = range(0,stop=T,length=N+1)
trajectories = [states[i][1, :] for i = 1:num_cars]
inputsworkers = [inputs[i] for i = 1:num_cars]

# Plot trajectories
trajectoryPlot = plot(t, trajectories, legend=false)
display(trajectoryPlot)

# Plot scaled error (discrepancy between equivalent agent paths according to different neighbourhood solutions)
scaled_error = violation_norm / violation_norm[1]
resplot = plot(1:iterations, scaled_error, xlab="Iterations", ylab="Scaled error", legend=false)

# Find input cost
inputCost = sum(sum([inputs[i] .* inputs[i] for i = 1:num_cars]))
println("Input cost = ", inputCost)
