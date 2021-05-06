
"""
Consensus algorithm and some required functions, applicable to formation problem
"""


include("FormationModel.jl")
include("DataTransfer.jl")





function consistency(model::Model, Z_dict::Dict, nhood::Array)

    X = vcat([model[:x][j] for j in nhood]...)
    Z = vcat([Z_dict[j] for j in nhood]...)

    @objective(model, Min, sum(sum([model[:u][j].^2 for j in nhood])) + 10 * sum((X-Z).^2))

    nothing

end











function solve_problem(model::Model, Z_dict::Dict, nhood::Array, stored_mean::Bool)

    stored_mean && consistency(model, Z_dict, nhood)

    optimise_model(model)
    X_dict, U_dict = Dict(), Dict()
    for j in nhood
        X_dict[j], U_dict[j] = value.(model[:x][j]), value.(model[:u][j])
    end

    return X_dict, U_dict

end















@everywhere function consensus(sys::Int, hub::Int, parameters::Tuple, neighbours::Array; 
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
        X_dict, U_dict = solve_problem(model, Z_dict, nhood, stored_mean)
        println("problem solved")

        # Gather assumed trajectories from neighbours, and broadcast trajectories to neigbours
        x_sys = x_exchange(X_dict, sys, neighbours, agent_procs)

        # Find mean of all trajectory predictions/assumptions for agent
        z = nhood_mean(x_sys, sys, nhood)
   
        # Gather z values from neighbours (and broadcast to neighbours) to construct Z
        Z_dict = z_exchange(z, sys, neighbours, agent_procs)

        SOLVED = false 
        ALL_SOLVED = hub_exchange(sys, hub, SOLVED, agent_procs, num)

        

        history[iteration] = (X_dict[sys], z)




        # STOP TIMING LOOP
        end



        println("Iteration ", iteration, " complete: ", loop)
        iteration += 1



    end


    # STOP TIMING TOTAL TIME IN WHILE LOOP
    end

    return X_dict[sys], U_dict[sys], history, totalloop


end