
"""
ADMM algorithm and some required functions, applicable to formation problem
"""

using Distributed
@everywhere using JuMP
@everywhere using Ipopt

@everywhere include("FormationModel.jl")






@everywhere function model_setup(sys::Int, parameters::Tuple, nhood::Array)

    _, _, N = parameters
    model = base_model(sys, parameters)
    update_model(model, nhood, [sys], parameters)

    return model

end









@everywhere function consistency(model::Model, Z_dict::Dict, nhood::Array, rho::Float64, lambda::Array)

    X = vcat([model[:x][j] for j in nhood]...)
    Z = vcat([Z_dict[j] for j in nhood]...)

    @objective(model, Min, sum(sum([model[:u][j].^2 for j in nhood])) + sum(lambda.*(X-Z)) + 0.5*rho*sum((X-Z).^2))

    nothing

end











@everywhere function solve_problem(model::Model, Z_dict::Dict, nhood::Array, stored_mean::Bool, rho::Float64, lambda::Array)

    stored_mean && consistency(model, Z_dict, nhood, rho, lambda)

    optimise_model(model)
    X_dict, U_dict = Dict(), Dict()
    for j in nhood
        X_dict[j], U_dict[j] = value.(model[:x][j]), value.(model[:u][j])
    end

    return X_dict, U_dict

end










# Upload sys assumptions of j, and fetch j assumptions of sys from j 

@everywhere function x_exchange(X_dict::Dict, sys::Int, neighbours::Array, agent_procs::Dict)

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









@everywhere function z_exchange(z::Array, sys::Int, neighbours::Array, agent_procs::Dict)

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








@everywhere function hub_exchange(sys::Int, hub::Int, solution::Array, agent_procs::Dict, parameters::Tuple)


    num, _ = parameters

    if sys != hub

	put!(to_hub, solution)
        remotecall_fetch(wait, agent_procs[hub], @spawnat(agent_procs[hub], from_hub))
        SOLVED = fetch(@spawnat(agent_procs[hub], take!(from_hub)))

    else

        agent_solution = Dict()
        agent_solution[hub] = solution


        @sync for j in filter(x->x!=hub, Array(1:num))
            @async begin
                remotecall_fetch(wait, agent_procs[j], @spawnat(agent_procs[j], to_hub))
                agent_solution[j] = fetch(@spawnat(agent_procs[j], take!(to_hub)))
            end
        end


        con_vals = [coupled_inequalities(agent_solution, pairing(Array(1:num)), parameters) for i = 1:num]
        SOLVED = maximum([maximum(con_vals[i]) for i = 1:num]) <= 0 ? true : false

        SOLVED = false

        println("SOLVED = ", SOLVED)
        for _ in 1:num-1
            put!(from_hub, SOLVED)
	end

    end

    return SOLVED

end







@everywhere function nhood_mean(dict::Dict, sys::Int, nhood::Array)

    agg = zeros(size(dict[nhood[1]]))
    for j in nhood
        agg += dict[j]
    end
    return agg / length(nhood)

end










@everywhere function ADMM(sys::Int, hub::Int, parameters::Tuple, neighbours::Array; rho::Float64 = 0.1,
                               agent_procs::Dict = Dict(i=>sort(workers())[i] for i = 1:parameters[1]))
                          

    # Define constant parameters 
    num, _, N = parameters
    nhood = [([sys, neighbours]...)...]


    # Create model
    model = model_setup(sys, parameters, nhood)
    

    # Initialise variables
    X = Array{Float64, 2}(undef, 4*length(nhood), N+1)
    Z = Array{Float64, 2}(undef, 4*length(nhood), N+1)
    X_dict, U_dict, Z_dict, history = Dict(), Dict(), Dict(), Dict()
    x_sys = Any[]
    lambda = zeros(4 * length(nhood), N+1)


    # Initialise channels for global communication
    if sys == hub
        global from_hub = Channel{Bool}(num-1)
    else
        global to_hub = Channel{Any}(1)
    end




    # BEGIN TIMING TOTAL TIME IN WHILE LOOP
    totalloop = @elapsed begin




    iteration = 1

    while iteration <= 20

        # BEGIN TIMING LOOP
        loop = @elapsed begin



        # Initialise channels
        global chz = Channel{Any}(1)
        for j in neighbours
            Core.eval(Main, Expr(:(=), Symbol("chx", j), Channel{Any}(1)))
        end


        # Solve problem 
        stored_mean = iteration == 1 ? false : true
        X_dict, U_dict = solve_problem(model, Z_dict, nhood, stored_mean, rho, lambda)

        # Gather assumed trajectories from neighbours, and broadcast trajectories to neigbours
        x_sys = x_exchange(X_dict, sys, neighbours, agent_procs)

        # Find mean of all trajectory predictions/assumptions for agent
        z = nhood_mean(x_sys, sys, nhood)
   
        # Gather z values from neighbours (and broadcast to neighbours) to construct Z
        Z_dict = z_exchange(z, sys, neighbours, agent_procs)

        # ADMM lambda update
        X = vcat([X_dict[j] for j in nhood]...)
        Z = vcat([Z_dict[j] for j in nhood]...)
        lambda = lambda + rho * (X-Z)

        solution = X_dict[sys]
        SOLVED = hub_exchange(sys, hub, solution, agent_procs, parameters)


        history[iteration] = (X_dict[sys], z)




        # STOP TIMING LOOP
        end



        println("Iteration ", iteration, " complete: ", loop)

        if SOLVED
            break
        end

        iteration += 1



    end


    # STOP TIMING TOTAL TIME IN WHILE LOOP
    end

    return X_dict[sys], U_dict[sys], history, totalloop


end