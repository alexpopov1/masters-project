"""
Centralised optimisation for vehicle platoon ring problem
"""


using JuMP     # Optimisation problem definition
using Ipopt    # Optimisation solver


include("RingModel.jl")




@everywhere function centralised(parameters)



    num_cars, _ = parameters

    model = base_model(1, parameters)
    update_model(model, Array(1:num_cars), [1], parameters)


    @constraint(model, coupled_inequalities(model[:x][num_cars], model[:x][1], parameters, true) .<= 0)

    @time begin
    optimise_model(model)

    states = Dict()
    inputs = Dict()
    for i = 1:num_cars
        states[i] = value.(model[:x][i])
        inputs[i] = value.(model[:u][i])
    end

    end
    return states, inputs

end
