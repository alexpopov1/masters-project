
using Distributed





@everywhere function initialise(sys::Int, parameters::Tuple)

    numCars, T, N, init = parameters

    states = Array{Any, 2}(undef, 4, N+1)
    states[:, 1] = init[sys]
    inputs = zeros(2, N)

    for k = 1:N
        states[1, k+1]  = states[1, k] + (T/N) * states[3, k] 
        states[2, k+1] = states[2, k] + (T/N) * states[4, k]
        states[3, k+1] = states[3, k]
        states[4, k+1] = states[4, k]
    end

    return states, inputs

end






@everywhere function equalities(sys::Int, x::Array, u::Array, parameters::Tuple)
    
    num, T, N, init, v_final  = parameters
    constraints = Array{Any, 1}(undef, 4N+6)

    constraints[1:N] = [x[1, k+1] - x[1, k] - (T/N) * x[3, k] - 0.5 * (T/N)^2 * u[1, k] for k = 1:N]
    constraints[N+1:2N] = [x[2, k+1] - x[2, k] - (T/N) * x[4, k] - 0.5 * (T/N)^2 * u[2, k] for k = 1:N]
    constraints[2N+1:3N] = [x[3, k+1] - x[3, k] - (T/N) * u[1, k] for k = 1:N]
    constraints[3N+1:4N] = [x[4, k+1] - x[4, k] - (T/N) * u[2, k] for k = 1:N]

    constraints[4N+1:4N+4] = x[:, 1] - init[sys]
    constraints[4N+5:4N+6] = x[3:4, N+1] - v_final


    return constraints

end





               

@everywhere function uncoupled_inequalities(x::Array, u::Array, parameters::Tuple)

    _, _, N, ~, ~, (umax, vmax, rmin, xc, yc, rc) = parameters

    constraints = Array{Any, 1}(undef, 2N+1)

    # Lower bound for control inputs
    constraints[1:N] = [u[1, k]^2 + u[2, k]^2 - umax for k = 1:N]

    # Maximum speed
    constraints[N+1:2N] = [(x[3, k] + x[4, k])^2 - vmax^2 for k = 1:N]

    # Terminal location range
    constraints[2N+1] = (x[1, N+1] - xc)^2 + (x[2, N+1] - yc)^2 - rc^2

    return constraints

end








@everywhere function coupled_inequalities(x::Dict, pairs::Array, parameters::Tuple)


     _, _, N, ~, ~, (~, ~, rmin, ~, ~, ~) = parameters
    constraints = Array{Any, 1}(undef, (N+1)*length(pairs))

    for i = 1:length(pairs)
        xA, xB = x[pairs[i][1]], x[pairs[i][2]]
        constraints[(N+1)*i-N:(N+1)*i] = [rmin^2 - (xA[1, k] - xB[1, k])^2 - (xA[2, k] - xB[2, k])^2 for k = 1:N+1]
    end

    return constraints

end








@everywhere function base_model(sys::Int, parameters::Tuple, iter_limit::Int = 1000)

    _, _, N = parameters

    model = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", iter_limit)

    x, u = Dict(), Dict()   
    x[sys] = @variable(model, [1:4, 1:N+1], base_name = "x$sys")
    u[sys] = @variable(model, [1:2, 1:N], base_name = "u$sys")
    

    model[:x] = x    # Bind variables...
    model[:u] = u    # ... to model

    @objective(model, Min, sum(u[sys].^2))
    @constraint(model, equalities(sys, x[sys], u[sys], parameters) .== 0)          # Equality constraints
    @constraint(model, uncoupled_inequalities(x[sys], u[sys], parameters) .<= 0)   # Uncoupled inequality constraints
    
    return model

end





@everywhere function pairing(array::Array)

    pairs = Any[]

    for i = 1:length(array)-1
        for j = i+1:length(array)
            push!(pairs, [array[i], array[j]])
        end
    end

    return pairs

end









@everywhere function update_model(model::Model, nhood::Array, 
                                  prev_nhood::Array, parameters::Tuple)


    if nhood == prev_nhood

        nothing

    else

        num, _, N = parameters
        new_neighbours = setdiff(nhood, prev_nhood)
        new_pairs = setdiff(pairing(nhood), pairing(prev_nhood))

        x = model[:x]
        u = model[:u]

        for j in new_neighbours
            x[j] = @variable(model, [1:4, 1:N+1], base_name = "x$j")
            u[j] = @variable(model, [1:2, 1:N], base_name = "u$j")
        end
    
        @objective(model, Min, sum(sum([u[j].^2 for j in nhood])))
        @constraint(model, [j in new_neighbours], equalities(j, x[j], u[j], parameters) .== 0)
        @constraint(model, [j in new_neighbours], uncoupled_inequalities(x[j], u[j], parameters) .<= 0)
        @constraint(model, coupled_inequalities(x, new_pairs, parameters) .<= 0)
              
        nothing

    end

end








@everywhere function optimise_model(model::Model)

    #set_silent(model)
    optimize!(model)
    # println(termination_status(model))
    nothing
   
end



