include("FormationModel.jl")
include("WarmStart.jl")


@everywhere function centralised(parameters)

    num, _ = parameters
    model = base_model(1, parameters)
    update_model(model, Array(1:num), [1], parameters)
    warm_start(model, parameters, Array(1:num), [])
    timing = @elapsed optimise_model(model)

    states = Dict()
    inputs = Dict()

    for i = 1:num
        states[i] = value.(model[:x][i])
        inputs[i] = value.(model[:u][i])
    end

    return states, inputs, timing

end
