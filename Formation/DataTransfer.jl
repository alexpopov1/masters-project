

""" 
A series of functions implementing the data transfer between processes for ADMM
"""

using Distributed
include("FormationModel.jl")







function hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, 
                      parameters::Tuple, new_agents::Array, fixed::Array, solve_time::Float64)


    num, _ = parameters


    if sys != hub

	put!(to_hub, (solution, new_agents, solve_time))
        remotecall_fetch(wait, agent_procs[hub], @spawnat(agent_procs[hub], from_hub))
        SOLVED, fixed = fetch(@spawnat(agent_procs[hub], take!(from_hub)))

    else
        pairs = pairing(Array(1:num))
        agent_solution, new_additions, timing = Dict(), Dict(), Dict()
        agent_solution[hub] = solution
        new_additions[hub] = new_agents
        timing[hub] = solve_time

        @sync for j in filter(x->x!=hub, Array(1:num))
            @async begin
                remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], to_hub))
                agent_solution[j], new_additions[j], timing[j] = fetch(@spawnat(agent_procs[j], take!(to_hub)))
            end
        end

        con_vals = [coupled_inequalities(agent_solution, pairs, parameters) for i = 1:num]

        SOLVED = maximum([maximum(con_vals[i]) for i = 1:num]) <= 1e-6 ? true : false

        fixed = isempty(fixed) ? [8] : union(fixed, union(vcat(values(new_additions)...)))
        # fixed = isempty(fixed) ? [findmax(timing)[2]] : union(fixed, union(vcat(values(new_additions)...)))
        # fixed = isempty(fixed) ? [Int(num/2)] : union(fixed, union(vcat(values(new_additions)...)))       
        if solve_time >= 0
            println("fixed = ", fixed)
        end

        println("SOLVED = ", SOLVED)
        for _ in 1:num-1
            put!(from_hub, (SOLVED, fixed))
	end

    end

    return SOLVED, fixed

end








function hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, parameters::Tuple)

    SOLVED, _ = hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, parameters::Tuple, [], [], -1.0)
    return SOLVED

end








function x_exchange(X_dict::Dict, sys::Int, neighbours::Array, agent_procs::Dict)

    x_sys = Dict()  
    x_sys[sys] = X_dict[sys]

    @sync for j in neighbours
        @async begin
            put!(getfield(Main, Symbol("chx", j)), X_dict[j])
            remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], getfield(Main, Symbol("chx", sys))))
            x_sys[j] = fetch(@spawnat(agent_procs[j], fetch(getfield(Main, Symbol("chx", sys)))))
        end
    end 

    return x_sys 
 
end







function z_exchange(z::Array, sys::Int, neighbours::Array, agent_procs::Dict)

    Z_dict = Dict()
    Z_dict[sys] = z

    put!(chz, z) 
    @sync for j in neighbours
        @async begin
            remotecall_fetch(wait, agent_procs[j], getfield(Main, :chz))
            Z_dict[j] = fetch(@spawnat(agent_procs[j], fetch(getfield(Main, :chz))))
        end
    end

    return Z_dict

end
