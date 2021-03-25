
using Distributed
@everywhere using JuMP
@everywhere using Ipopt
@everywhere using ParallelDataTransfer

@everywhere include("DataTransferTools.jl")
@everywhere include("RingModel.jl")
@everywhere include("C:/Users/apbab/OneDrive/Documents/Year 4/FYP/Julia Scripts/External Active Set/warmStart.jl")
warm_start = warmStart











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
            








@everywhere function solve_problem(model::Model, sys::Int, nodes::Array, prev_nodes::Array,
                                   parameters::Tuple, iter_limit::Int)

    update_model(model, nodes, prev_nodes, parameters)
    
    if nodes == prev_nodes 
        iter_limit *= 2
        set_optimizer_attribute(model, "max_iter", iter_limit)     
    end

    if has_values(model)
        for i in prev_nodes
            warm_start(model[:x][i], model[:u][i], value.(model[:x][i]), value.(model[:u][i]))
        end
        for i in setdiff(nodes, prev_nodes)
            states, inputs = initialise(i, parameters)
            warm_start(model[:x][i], model[:u][i], states, inputs)
        end
    else
        for i in nodes
            states, inputs = initialise(i, parameters)
            warm_start(model[:x][i], model[:u][i], states, inputs)
        end
    end


    optimise_model(model)                                         
    return value.(model[:x][sys]), value.(model[:u][sys]), iter_limit  

end









"""

DATA EXCHANGE BETWEEN AGENTS:

The current agent (sys) needs to fill two dictionaries: node_solutions and nodes_of_nodes. These will
be filled with the appropriate data from the agent's neighbourhood, so first it can fill in its own
data (key = sys). 

The agent has to make its data available to any other agents that require it, so it uploads its 
solutions and neighbourhood index set to the channel c. 

To get information from neighbours, the agent calls a wait function at each neighbour, waiting for the
local c channel to contain data. Once there is data on a neighbour's c, sys will fetch this data and 
store it in the dictionaries with the appropriate key.

"""



@everywhere function neighbour_exchange(sys::Int, opt_states::Array, nodes::Array, neighbours::Array, agent_procs::Dict)

        node_solutions = Dict()
        node_solutions[sys] = opt_states

        nodes_of_nodes = Dict()
        nodes_of_nodes[sys] = nodes

        put!(c, (opt_states, nodes)) 
 
        @sync for j in neighbours
            @async begin 
                remotecall_fetch(wait, agent_procs[j], getfield(Main, :c))
                node_solutions[j], nodes_of_nodes[j] = fetch(@spawnat(agent_procs[j], fetch(getfield(Main, :c))))
            end
        end

        return node_solutions, nodes_of_nodes

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










@everywhere function update_neighbourhood(nodes::Array, nodes_of_nodes::Dict, colliding_pairs::Array) 
    
    for i in unique([(colliding_pairs...)...])
        union!(nodes, nodes_of_nodes[i])                      
    end

    return ordering(nodes) 

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
    prev_nodes = [sys]
    nodes = ordering([([sys, neighbours]...)...])
    model = base_model(sys, parameters, iter_limit)
    opt_states, opt_inputs = Array{Float64, 2}, Array{Float64, 2}]
    it = 1


    # Initialise channels
    global c = Channel{Any}(1) 
    if sys == hub
        global from_hub = Channel{Bool}(num_cars-1)
    else
        global to_hub = Channel{Bool}(1)
    end
    

    while true

        # Solve problem
        println(sys, ": ", nodes) 
	opt_states, opt_inputs, iter_limit = solve_problem(model, sys, nodes, prev_nodes, parameters, iter_limit)


        # Upload data and receive neighbour data
        node_solutions, nodes_of_nodes = neighbour_exchange(sys, opt_states, nodes, neighbours, agent_procs)


        # Identify collisions between consecutive agents in neighbourhood
	ordered_solutions = [node_solutions[j] for j in nodes]
	colliding_pairs = collisions(ordered_solutions, nodes, parameters)


        # Check for globally feasible solution
        SOLVED = isempty(colliding_pairs) ? true : false
        ALL_SOLVED = hub_exchange(sys, hub, SOLVED, agent_procs, num_cars)
        if ALL_SOLVED 
            break
        end


        # Reset local channel
        global c = Channel{Any}(1)


        # Update graph
        prev_nodes = copy(nodes)
        nodes = update_neighbourhood(nodes, nodes_of_nodes, colliding_pairs)
	neighbours = filter(x->x!=sys, nodes)

        
        it += 1

    end  

    return opt_states, opt_inputs, (nodes, prev_nodes)

end



                   
 






































































