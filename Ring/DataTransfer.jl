
using Distributed
include("RingModel.jl")





function hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, parameters::Tuple)


    num_cars, _ = parameters


    if sys != hub

	put!(to_hub, solution)
        remotecall_fetch(wait, agent_procs[hub], @spawnat(agent_procs[hub], from_hub))
        SOLVED = fetch(@spawnat(agent_procs[hub], take!(from_hub)))

    else

        agent_solution = Dict()
        agent_solution[hub] = solution


        @sync for j in filter(x->x!=hub, Array(1:num_cars))
            @async begin
                remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], to_hub))
                agent_solution[j] = fetch(@spawnat(agent_procs[j], take!(to_hub)))
            end
        end

        con_vals = [coupled_inequalities(agent_solution[i], agent_solution[i%num_cars+1],
                          parameters, i == num_cars ? true : false) for i = 1:num_cars]

        SOLVED = maximum([maximum(con_vals[i]) for i = 1:num_cars]) <= 0 ? true : false


        println("SOLVED = ", SOLVED)
        for _ in 1:num_cars-1
            put!(from_hub, SOLVED)
	end

    end

    return SOLVED

end








# For consensus and ADMM algorithms
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







# For new algorithm
function x_exchange(X_dict::Dict, U_dict::Dict, sys::Int, neighbours::Array, agent_procs::Dict)

    x_from, u_from = Dict(), Dict()  
    x_from[sys], u_from[sys] = X_dict, U_dict

    @sync for j in neighbours
        @async begin
            put!(getfield(Main, Symbol("chx", j)), (X_dict, U_dict))
            remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], getfield(Main, Symbol("chx", sys))))
            x_from[j], u_from[j] = fetch(@spawnat(agent_procs[j], fetch(getfield(Main, Symbol("chx", sys)))))
        end
    end 

    return x_from, u_from
 
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





