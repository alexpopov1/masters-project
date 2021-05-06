

"""
Alternative consensus algorithm and some required functions, applicable to formation problem
"""



include("FormationModel.jl")
include("DataTransfer.jl")




function reduced_model(X_fixed::Dict, nhood::Array, parameters::Tuple)

    num, _, N = parameters

    if length(X_fixed) == length(nhood)

        model = Model()

    else 

        fixed_agents = keys(X_fixed)
        variable_agents = setdiff(nhood, fixed_agents)

        model = base_model(variable_agents[1], parameters)
        update_model(model, variable_agents, [variable_agents[1]], parameters)

        x, u = model[:x], model[:u]

        mixed_pairs = []
        pairs = pairing(nhood)
        for pair in pairs
            if (pair[1] in fixed_agents && pair[2] in variable_agents) ||
               (pair[1] in variable_agents && pair[2] in fixed_agents)
                push!(mixed_pairs, pair)
            end
        end

        vars_and_constants = Dict()
        for i in nhood
            if i in fixed_agents
                vars_and_constants[i] = X_fixed[i]
            else
                vars_and_constants[i] = x[i]
            end
        end

        new_cons = coupled_inequalities(vars_and_constants, mixed_pairs, parameters


        @constraint(model, vcat(new_cons...) .<= 0)
        

    end 
    

    return model


end







function solve_problem(model::Model, X_fixed::Dict, U_fixed::Dict, nhood::Array, stored_mean::Bool, sys::Int)

    if length(X_fixed) < length(nhood)
        optimise_model(model)
    end

    X_dict, U_dict = Dict(), Dict()

    for j in nhood
        X_dict[j], U_dict[j] = j in keys(X_fixed) ? (X_fixed[j], U_fixed[j]) : (value.(model[:x][j]), value.(model[:u][j]))
    end

    return X_dict, U_dict

end




















function algorithm(sys::Int, hub::Int, parameters::Tuple, neighbours::Array; 
                   agent_procs::Dict = Dict(i=>sort(workers())[i] for i = 1:parameters[1]), max_iterations = 10)
                          

    # Define constant parameters 
    num, _, N = parameters
    nhood = [([sys, neighbours]...)...]


    # Create model
    model = model_setup(sys, parameters, nhood)
    

    # Initialise variables
    X_dict, U_dict, X_fixed, U_fixed, Z_dict, history = Dict(), Dict(), Dict(), Dict(), Dict(), Dict()
    x_sys = Any[]




    if sys == hub
        global from_hub = Channel{Any}(num-1)
    else
        global to_hub = Channel{Any}(1)
    end


    subgraph = []
    fixed = []


    # BEGIN TIMING TOTAL TIME IN WHILE LOOP
    totalloop = @elapsed begin



    iteration = 1

    while iteration <= max_iterations

        # BEGIN TIMING LOOP
        loop = @elapsed begin


        # Initialise channels
        global chz = Channel{Any}(1)
        for j in neighbours
            Core.eval(Main, Expr(:(=), Symbol("chx", j), Channel{Any}(1)))
        end
       
        println("fixed: ", fixed)
        # Solve problem 
        stored_mean = iteration == 1 ? false : true
        solve_time = @elapsed X_dict, U_dict = solve_problem(model, X_fixed, U_fixed, nhood, stored_mean, sys)   


        # Gather assumed trajectories from neighbours, and broadcast trajectories to neigbours
        x_from, u_from = x_exchange(X_dict, U_dict, sys, neighbours, agent_procs)

        # Check global solution via hub agent
        solution = X_dict[sys]
        new_agents = sys in fixed ? setdiff(nhood, fixed) : []
        
        SOLVED, fixed = hub_exchange(sys, hub, solution, agent_procs, parameters, new_agents, fixed, solve_time)
        
        if SOLVED
            println("Iteration ", iteration, " complete")
            break
        end




        X_fixed = Dict()

  
        
        if length(intersect(fixed, nhood)) >= 1
            for j in nhood
                for k in intersect(fixed, nhood)
                    if j in keys(x_from[k])
                        X_fixed[j], U_fixed[j] = x_from[k][j], u_from[k][j]
                    end
                end     
            end  
            model = reduced_model(X_fixed, nhood, parameters)  
        end  
  

        history[iteration] = X_dict[sys]




        # STOP TIMING LOOP
        end 


        println("Iteration ", iteration, " complete: ", loop)
        iteration += 1


    end


    # STOP TIMING TOTAL TIME IN WHILE LOOP
    end

    return X_dict[sys], U_dict[sys], history, totalloop, X_dict


end