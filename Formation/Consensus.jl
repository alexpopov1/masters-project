
"""
Consensus algorithm and some required functions, applicable to formation problem
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









@everywhere function consistency(model::Model, Z_dict::Dict, nhood::Array, stored_cons::Bool)

    X = vcat([model[:x][j][1, :] for j in nhood]...)
    Z = vcat([Z_dict[j][1, :] for j in nhood]...)


    if stored_cons

        for j = 1:length(model[:lower])
            set_normalized_rhs(model[:lower][j], Z[j] - 0)
            set_normalized_rhs(model[:upper][j], Z[j] + 0)
        end
        
    else

        lower = @constraint(model, X .>= Z .- 0)
        upper = @constraint(model, X .<= Z .+ 0)
        model[:lower] = lower
        model[:upper] = upper
        
    end

    nothing

end











@everywhere function solve_problem(model::Model, Z_dict::Dict, nhood::Array, stored_mean::Bool, stored_cons::Bool)

    stored_mean && consistency(model, Z_dict, nhood, stored_cons)

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










@everywhere function nhood_mean(dict::Dict, sys::Int, nhood::Array)

    agg = zeros(size(dict[nhood[1]]))
    for j in nhood
        agg += dict[j]
    end
    return agg / length(nhood)

end










@everywhere function consensus(sys::Int, parameters::Tuple, neighbours::Array; 
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



    iteration = 1

    while iteration <= 2

       
        # Initialise channels
        global chz = Channel{Any}(length(neighbours))
        for j in neighbours
            Core.eval(Main, Expr(:(=), Symbol("chx", j), Channel{Any}(1)))
        end


        # Solve problem 
        stored_mean = iteration == 1 ? false : true
        stored_cons = iteration <= 2 ? false : true
        X_dict, U_dict = solve_problem(model, Z_dict, nhood, stored_mean, stored_cons)
        println("problem solved")

        # Gather assumed trajectories from neighbours, and broadcast trajectories to neigbours
        x_sys = x_exchange(X_dict, sys, neighbours, agent_procs)
        println("finish x exchange")

        # Find mean of all trajectory predictions/assumptions for agent
        z = nhood_mean(x_sys, sys, nhood)
   
        # Gather z values from neighbours (and broadcast to neighbours) to construct Z
        Z_dict = z_exchange(z, sys, neighbours, agent_procs)
        println("finish z exchange")

        history[iteration] = X_dict[sys]
        println("Iteration ", iteration, " complete")
   
        iteration += 1



    end



    return X_dict[sys], U_dict[sys], history


end
