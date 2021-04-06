
"""
Smallest Neighbourhood algorithm and some required functions, applicable to vehicle 
platoon ring problem

"""


using Distributed         # Distributed implementation
@everywhere using JuMP    # Optimisation problem definition
@everywhere using Ipopt   # Optimisation solver

@everywhere include("RingModel.jl")









@everywhere function eval_collision_constraints(states_array::Array, indices::Array, parameters::Tuple)

    num_cars, _ = parameters
    return [coupled_inequalities(states_array[i], states_array[i+1], 
            parameters, indices[i] == num_cars ? true : false) 
            for i = 1:length(states_array)-1]

end











@everywhere function collisions(states_array::Array, indices::Array, parameters::Tuple)

    pairs= []
    vals = eval_collision_constraints(states_array, indices, parameters)
    # println("max value: ", maximum([maximum(vals[i]) for i = 1:length(vals)]))
    for i = 1:length(vals)
        if maximum(vals[i]) > 0
            push!(pairs, (indices[i], indices[i+1]))
        end
    end

    return pairs

end
            








@everywhere function warm_start(x::Array, u::Array, states::Array, inputs::Array)

    # Set warm start for states
    for j = 1:size(x)[1]
        for k = 1:size(x)[2]
            set_start_value(x[j,k], states[j,k])       
        end
    end


    # Set warm start for inputs
    if ndims(u) == 1

        for j = length(u)
            set_start_value(u[j], inputs[j])    
        end

    else        
    
        for j = 1:size(u)[1]
            for k = 1:size(u)[2]
                set_start_value(u[j,k], inputs[j,k])
            end
        end

    end

end










@everywhere function warm_start(model::Model, parameters::Tuple, nhood::Array, prev_nhood::Array)

    if has_values(model)

        for i in prev_nhood
            warm_start(model[:x][i], model[:u][i], value.(model[:x][i]), value.(model[:u][i]))
        end

        for i in setdiff(nhood, prev_nhood)
            states, inputs = initialise(i, parameters)
            warm_start(model[:x][i], model[:u][i], states, inputs)
        end

    else

        for i in nhood
            states, inputs = initialise(i, parameters)
            warm_start(model[:x][i], model[:u][i], states, inputs)
        end

    end

end







@everywhere function solve_problem(model::Model, sys::Int, nhood::Array, prev_nhood::Array,
                                   parameters::Tuple, iter_limit::Int)


    if nhood == prev_nhood
        iter_limit *= 2
        set_optimizer_attribute(model, "max_iter", iter_limit)     
    end


    update = @elapsed update_model(model, nhood, prev_nhood, parameters)
    warmstart = @elapsed warm_start(model, parameters, nhood, prev_nhood)
    solving = @elapsed optimise_model(model)  
    println("update: ", update, "   warmstart: ", warmstart, "   solver: ", solving)                                       
    @time return value.(model[:x][sys]), value.(model[:u][sys]), iter_limit  

end









"""

DATA EXCHANGE BETWEEN AGENTS:

The current agent (sys) needs to fill two dictionaries: nhood_solutions and nhood_of_agents. These will
be filled with the appropriate data from the agent's neighbourhood, so first it can fill in its own
data (key = sys). 

The agent has to make its data available to any other agents that require it, so it uploads its 
solutions and neighbourhood index set to the channel c. 

To get information from neighbours, the agent calls a wait function at each neighbour, waiting for the
local c channel to contain data. Once there is data on a neighbour's c, sys will fetch this data and 
store it in the dictionaries with the appropriate key.

"""



@everywhere function neighbour_exchange(sys::Int, opt_states::Array, nhood::Array, neighbours::Array, agent_procs::Dict)

        nhood_solutions = Dict()
        nhood_solutions[sys] = opt_states

        nhood_of_agents = Dict()
        nhood_of_agents[sys] = nhood

        put!(c, (opt_states, nhood)) 
 
        @sync for j in neighbours
            @async begin 
                remotecall_fetch(wait, agent_procs[j], getfield(Main, :c))
                nhood_solutions[j], nhood_of_agents[j] = fetch(@spawnat(agent_procs[j], fetch(getfield(Main, :c))))
            end
        end

        return nhood_solutions, nhood_of_agents

end










"""

DATA EXCHANGE WITH HUB:

A non-hub agent will upload SOLVED boolean to its to_hub channel, then remotely call a wait function on 
the hub agent, to wait for the channel from_hub to contain data. Once this channel contains data, the 
data (ALL_SOLVED boolean) will be taken by the agent, telling it whether the problem has been
globally solved.

The hub agent calls a wait function at each other agent in the system, to wait for the local to_hub to 
contain data. Once there is data on a channel (SOLVED boolean), the hub will take it. Once all SOLVED
booleans have been taken, the hub determines the value of ALL_SOLVED, and then uploads it to from_hub.

"""



@everywhere function hub_exchange(sys::Int, hub::Int, SOLVED::Bool, agent_procs::Dict, num_cars::Int)


	if sys != hub
          
	    put!(to_hub, SOLVED)                  
            remotecall_fetch(wait, agent_procs[hub], @spawnat(agent_procs[hub], from_hub))
            ALL_SOLVED = fetch(@spawnat(agent_procs[hub], take!(from_hub)))

	else

            agent_check = Dict()
            agent_check[hub] = SOLVED

            @sync for j in filter(x->x!=hub, Array(1:num_cars)) 
                @async begin
                    remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], to_hub))  
                    agent_check[j] = fetch(@spawnat(agent_procs[j], take!(to_hub))) 
                end
            end

            ALL_SOLVED = false in [agent_check[j] for j = 1:num_cars] ? false : true
            println("ALL_SOLVED = ", ALL_SOLVED)
            for _ in 1:num_cars-1
                put!(from_hub, ALL_SOLVED)
	    end

	end

        return ALL_SOLVED

end










@everywhere function update_neighbourhood(nhood::Array, nhood_of_agents::Dict, colliding_pairs::Array) 
    
    for i in unique([(colliding_pairs...)...])
        union!(nhood, nhood_of_agents[i])                      
    end

    return ordering(nhood) 

end










"""

Smallest neighbourhood algorithm solves a system problem, then compares optimal solutions from itself and
from the neighbourhood controllers. If any coupled constraints (in this case, collision avoidance) are
violated among these solutions, then identify the critical colliding pair and update the system neighbourhood
to include the full neighbourhoods of the pair.

""" 


@everywhere function smallest_neighbourhood(sys::Int, hub::Int, parameters::Tuple, neighbours::Array;
                                            agent_procs::Dict = Dict(i=>sort(workers())[i] for i = 1:parameters[1]),
                                            iter_limit::Int=1000)

    # Initialise variables
    num_cars, _ = parameters
    prev_nhood = [sys]
    nhood = ordering([([sys, neighbours]...)...])
    model = base_model(sys, parameters, iter_limit)
    opt_states, opt_inputs = Array{Float64, 2}, Array{Float64, 2}
    it = 1


    # Initialise channels
    global c = Channel{Any}(1) 
    if sys == hub
        global from_hub = Channel{Bool}(num_cars-1)
    else
        global to_hub = Channel{Bool}(1)
    end
    

    while true

        loop = @elapsed begin

        # Solve problem
        println(sys, ": ", nhood) 
	prob = @elapsed opt_states, opt_inputs, iter_limit = solve_problem(model, sys, nhood, prev_nhood, parameters, iter_limit)


        # Upload data and receive neighbour data
        nhood_solutions, nhood_of_agents = neighbour_exchange(sys, opt_states, nhood, neighbours, agent_procs)


        # Identify collisions between consecutive agents in neighbourhood
	ordered_solutions = [nhood_solutions[j] for j in nhood]
	colliding_pairs = collisions(ordered_solutions, nhood, parameters)


        # Check for globally feasible solution
        SOLVED = isempty(colliding_pairs) ? true : false
        ALL_SOLVED = hub_exchange(sys, hub, SOLVED, agent_procs, num_cars)
        if ALL_SOLVED 
            break
        end


        # Reset local channel
        global c = Channel{Any}(1)


        # Update graph
        prev_nhood = copy(nhood)
        nhood = update_neighbourhood(nhood, nhood_of_agents, colliding_pairs)
	neighbours = filter(x->x!=sys, nhood)

        
        it += 1


        end

        println("prob: ", prob, ", loop: ", loop)
    end  

    return opt_states, opt_inputs, (nhood, prev_nhood)

end



                   
 






































































