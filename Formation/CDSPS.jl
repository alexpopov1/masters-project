"""
Connected Dominating Set Path Sweep (CDS-PS) and some required functions, applicable to formation problem
"""


include("FormationModel.jl")
include("WarmStart.jl")



# Define reduced model by replacing already-solved local problems with fixed solutions
function new_model(X_fixed::Dict, nhood::Array, parameters::Tuple)

    num, _, N = parameters

    fixed_agents = keys(X_fixed)
    variable_agents = setdiff(nhood, fixed_agents)

    if isempty(variable_agents)

        model = Model()

    else

        agents = union(fixed_agents, variable_agents)

        model = base_model(variable_agents[1], parameters)
        update_model(model, variable_agents, [variable_agents[1]], parameters)
        x, u = model[:x], model[:u]

        mixed_pairs = []

        pairs = pairing(collect(agents))


        for pair in pairs
            if (pair[1] in fixed_agents && pair[2] in variable_agents) ||
               (pair[1] in variable_agents && pair[2] in fixed_agents)
                push!(mixed_pairs, pair)
            end
        end


        vars_and_constants = Dict()
        for i in agents
            if i in fixed_agents
                vars_and_constants[i] = X_fixed[i]
            else
                vars_and_constants[i] = x[i]
            end
        end

        new_cons = coupled_inequalities(vars_and_constants, mixed_pairs, parameters)

        @constraint(model, vcat(new_cons...) .<= 0)

    end

    return model

end







# Function to solve the optimisation problem according to its current form
function solve_problem(model::Model, x_fixed::Dict, u_fixed::Dict, 
                       nhood::Array, parameters::Tuple, prev_solutions::Dict)

    fixed_agents = keys(x_fixed)
    variable_agents = setdiff(nhood, fixed_agents)
    # println("variables: ", variable_agents, " fixed: ", fixed_agents)
    if length(variable_agents) >= 1
  

        if length(intersect(fixed_agents, nhood)) == 0
            warm_start(model, parameters, nhood)
        else
            warm_start(model, variable_agents, prev_solutions)
        end

        optimise_model(model)

    end

    X_dict, U_dict = Dict(), Dict()

    for j in nhood
        X_dict[j], U_dict[j] = j in fixed_agents ? (x_fixed[j], u_fixed[j]) : (value.(model[:x][j]), value.(model[:u][j]))
    end

    return X_dict, U_dict

end









# Implement CDS Path Sweep algorithm
function CDSPS(sys::Int, parameters::Tuple, neighbours::Array, path::Array;
                    agent_procs::Dict = Dict(i=>sort(workers())[i] for i = 1:parameters[1]))

    # Define constant parameters
    num, _, N = parameters
    nhood = [([sys, neighbours]...)...]

    # Create model
    model = model_setup(sys, parameters, nhood)

    # Initialise variables
    x_solution, u_solution = Array{Float64, 2}(undef, 4, N+1), Array{Float64, 2}(undef, 2, N+1)
    NODE, SOLVER, PASSER, FETCHER = false, false, false, false

    # Initialise communication
    if sys in path

        SOLVER = true
        position = findall(x->x==sys, path)[1]
        NODE = position == 1 ? true : false
        sender = position == 1 ? 0 : agent_procs[path[position-1]]

        x_fixed, u_fixed, new_x_fixed, new_u_fixed, prev_solutions = Dict(), Dict(), Dict(), Dict(), Dict()

        receivers = setdiff(nhood, path)
        for j in receivers
            Core.eval(Main, Expr(:(=), Symbol("chx", j), Channel{Any}(1)))
        end

        if position < length(path)
            PASSER = true
            global relay = Channel{Any}(1) 
        end

    else

        FETCHER = true
        links = intersect(nhood, path)
        order = Dict(i=>findall(x->x==i, path)[1] for i in links)
        sender = agent_procs[path[minimum(values(order))]]

    end
    

    # BEGIN TIMING TOTAL TIME IN WHILE LOOP
    totalloop = @elapsed begin
    iteration = 1
    while true

        # BEGIN TIMING TOTAL TIME IN SINGLE ITERATION
        loop = @elapsed begin

        if SOLVER    # solve problem with unfixed variables of neighbourhood and all fixed solutions
            X_dict, U_dict = solve_problem(model, x_fixed, u_fixed, nhood, parameters, prev_solutions)
        end
        

        # if node, pass all new fixed solutions forward in path (and check if solved - for testing purposes)
        if SOLVER
      
            if NODE
             
                for j in setdiff(nhood, keys(x_fixed))
                    x_fixed[j], u_fixed[j] = X_dict[j], U_dict[j]
                    new_x_fixed[j], new_u_fixed[j] = X_dict[j], U_dict[j]
                end
                global checker = x_fixed
                println("max = ", maximum(coupled_inequalities(x_fixed, pairing(collect(keys(x_fixed))), parameters)))
                for j in receivers
                    put!(getfield(Main, Symbol("chx", j)), (X_dict[j], U_dict[j]))
                end

                if PASSER
                    put!(relay, (new_x_fixed, new_u_fixed, true))
                end

                x_solution, u_solution = X_dict[sys], U_dict[sys]
                break

            else

                remotecall_fetch(wait, sender, @spawnat(sender, getfield(Main, :relay)))

                new_x_fixed, new_u_fixed, NODE = fetch(@spawnat(sender, take!(getfield(Main, :relay))))

                for j in keys(new_x_fixed)
                    x_fixed[j] = new_x_fixed[j]
                end
             
                for j in keys(new_u_fixed)
                    u_fixed[j] = new_u_fixed[j]
                end
                
                if iteration == 1
                    prev_solutions = Dict()
                    for j in setdiff(nhood, keys(x_fixed))
                        prev_solutions[j] = (X_dict[j], U_dict[j])
                    end
                end
                
                if PASSER
                    put!(relay, (new_x_fixed, new_u_fixed, false))
                end


            end

            model = new_model(x_fixed, nhood, parameters)

        elseif FETCHER
            
            remotecall_fetch(wait, sender, @spawnat(sender, getfield(Main, Symbol("chx", sys))))
            x_solution, u_solution = fetch(@spawnat(sender, take!(getfield(Main, Symbol("chx", sys)))))
            break

        end

        # END TIMING TOTAL TIME IN SINGLE ITERATION
        end

        println("Iteration ", iteration, " complete: ", loop) 
        iteration += 1

    end

    # STOP TIMING TOTAL TIME IN WHILE LOOP
    end


    return x_solution, u_solution, totalloop

end
