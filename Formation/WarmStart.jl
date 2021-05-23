include("FormationModel.jl")





function warm_start(x::Array, u::Array, states::Array, inputs::Array)

    # Set warm start for states
    for j = 1:size(x)[1]
        for k = 1:size(x)[2]
            set_start_value(x[j,k], states[j,k])
        end
    end


    # Set warm start for inputs
    if ndims(u) == 1

        for j = length(u)
            set_start_value(u[j], inputs[j])
        end

    else

        for j = 1:size(u)[1]
            for k = 1:size(u)[2]
                set_start_value(u[j,k], inputs[j,k])
            end
        end

    end

end










function warm_start(model::Model, parameters::Tuple, nhood::Array, prev_nhood::Array)

    if has_values(model)

        for i in prev_nhood
            warm_start(model[:x][i], model[:u][i], value.(model[:x][i]), value.(model[:u][i]))
        end

        for i in setdiff(nhood, prev_nhood)
            states, inputs = initialise(i, parameters)
            warm_start(model[:x][i], model[:u][i], states, inputs)
        end

    else

        for i in nhood
            states, inputs = initialise(i, parameters)
            warm_start(model[:x][i], model[:u][i], states, inputs)
        end

    end

end



function warm_start(model::Model, parameters::Tuple, nhood::Array)

    warm_start(model, parameters, nhood, nhood)

end




function warm_start(model::Model, variable_agents::Array, prev_solutions::Dict)

    for i in variable_agents
        x_solution, u_solution = prev_solutions[i]
        warm_start(model[:x][i], model[:u][i], x_solution, u_solution)
    end

end





