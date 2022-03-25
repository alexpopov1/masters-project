
"""
A series of functions implementing data transfer for ADMM and CDSBS
"""

using Distributed
include("PlatoonModel.jl")







function hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, 
                      parameters::Tuple, new_agents::Array, fixed::Array, solve_time::Float64)


    num_cars, _ = parameters


    if sys != hub

	put!(to_hub, (solution, new_agents, solve_time))
        remotecall_fetch(wait, agent_procs[hub], @spawnat(agent_procs[hub], from_hub))
        SOLVED, fixed = fetch(@spawnat(agent_procs[hub], take!(from_hub)))

    else

        agent_solution, new_additions, timing = Dict(), Dict(), Dict()
        agent_solution[hub] = solution
        new_additions[hub] = new_agents
        timing[hub] = solve_time

        @sync for j in filter(x->x!=hub, Array(1:num_cars))
            @async begin
                remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], to_hub))
                agent_solution[j], new_additions[j], timing[j] = fetch(@spawnat(agent_procs[j], take!(to_hub)))
            end
        end

        con_vals = [coupled_inequalities(agent_solution[i], agent_solution[i%num_cars+1],
                          parameters, i == num_cars ? true : false) for i = 1:num_cars]

        SOLVED = maximum([maximum(con_vals[i]) for i = 1:num_cars]) <= 1e-6 ? true : false
      
        # fixed = isempty(fixed) ? [findmax(timing)[2]] : union(fixed, union(vcat(values(new_additions)...)))
        fixed = isempty(fixed) ? [Int(num_cars/2)] : union(fixed, union(vcat(values(new_additions)...)))       
        if solve_time >= 0
            println("fixed = ", fixed)
        end

        println("SOLVED = ", SOLVED)
        for _ in 1:num_cars-1
            put!(from_hub, (SOLVED, fixed))
	end

    end

    return SOLVED, fixed

end








function hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, parameters::Tuple)

    SOLVED, _ = hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, parameters::Tuple, [], [], -1.0)
    return SOLVED

end








# For ADMM 
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







# For CDSBS
function x_exchange(X_dict::Dict, U_dict::Dict, sys::Int, neighbours::Array, agent_procs::Dict)

    x_from, u_from = Dict(), Dict()  
    x_from[sys], u_from[sys] = X_dict, U_dict

    @sync for j in neighbours
        @async begin
            put!(getfield(Main, Symbol("chx", j)), (X_dict, U_dict))
            remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], getfield(Main, Symbol("chx", sys))))
            x_from[j], u_from[j],  = fetch(@spawnat(agent_procs[j], fetch(getfield(Main, Symbol("chx", sys)))))
        end
    end 

    return x_from, u_from
 
end






# For ADMM
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



