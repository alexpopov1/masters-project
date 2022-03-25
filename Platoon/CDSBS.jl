
"""
Connected Dominating Set Branch Sweep algorithm and some required functions, applicable to vehicle platoon problem
"""



include("PlatoonModel.jl")
include("DataTransfer.jl")



# Define reduced model by treating already-solved local problems as fixed constraints for remaining agents
function reduced_model(X_fixed::Dict, nhood::Array, parameters::Tuple)

    num_cars, _, N = parameters

    if length(X_fixed) == length(nhood)

        model = Model()

    else 

        fixed_agents = keys(X_fixed)
        variable_agents = setdiff(nhood, fixed_agents)

        model = base_model(variable_agents[1], parameters)

        update_model(model, variable_agents, [variable_agents[1]], parameters)

        x, u = model[:x], model[:u]
        new_cons = []

        for i = 1:length(nhood)-1
            if nhood[i] in fixed_agents && nhood[i+1] in variable_agents
                push!(new_cons, 
                      coupled_inequalities(X_fixed[nhood[i]], x[nhood[i+1]], parameters, nhood[i]==num_cars ? true : false))
            elseif nhood[i] in variable_agents && nhood[i+1] in fixed_agents
                push!(new_cons,
                      coupled_inequalities(x[nhood[i]], X_fixed[nhood[i+1]], parameters, nhood[i]==num_cars ? true : false))
            end
        end

        @constraint(model, vcat(new_cons...) .<= 0)
        

    end 
    

    return model


end






# Solve optimisation problem according to current form
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



















# Implement CDSBS algorithm
function CDSBS(sys::Int, hub::Int, parameters::Tuple, neighbours::Array; 
                   agent_procs::Dict = Dict(i=>sort(workers())[i] for i = 1:parameters[1]), max_iterations = 10)
                          

    # Define constant parameters 
    num_cars, _, N = parameters
    nhood = ordering([([sys, neighbours]...)...])


    # Create model
    model = model_setup(sys, parameters, nhood)
    

    # Initialise variables
    X = Array{Float64, 2}(undef, 2*length(nhood), N+1)
    Z = Array{Float64, 2}(undef, 2*length(nhood), N+1)
    X_dict, U_dict, X_fixed, U_fixed, Z_dict, history = Dict(), Dict(), Dict(), Dict(), Dict(), Dict()
    x_sys = Any[]




    if sys == hub
        global from_hub = Channel{Any}(num_cars-1)
    else
        global to_hub = Channel{Any}(1)
    end



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
