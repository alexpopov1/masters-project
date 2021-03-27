



using Distributed





@everywhere function initialise(sys::Int, parameters::Tuple)

    numCars, T, N, _, _, (omega1, omega2), astart = parameters

    states = Array{Float64, 2}(undef, 2, N+1)
    states[1, 1] = sum(astart[1:sys])
    states[2, 1] = omega1[sys]
    inputs = zeros(N)

    for k = 1:N
        states[1, k+1]  = states[1, k] + (T/N) * states[2, k]
        states[2, k+1] = states[2, k] + (T/N) * inputs[k]
    end

    return states, inputs

end







@everywhere function ordering(nodes::Array)

    order = sort(nodes)
    for i = 1:length(order)-1
	if order[i+1] - order[i] != 1
	    order = [([order[i+1:length(order)], order[1:i]]...)...]
            break
	end
    end   

    return order

end









@everywhere function ordering_new_neighbourhood(nodes::Array{Int, 1}, prev_nodes::Array{Int, 1}, parameters::Tuple)

    _, _, N = parameters
    
    rear_index = findall(node->node==prev_nodes[1], nodes)[1]
    front_index = findall(node->node==last(prev_nodes), nodes)[1]

    if rear_index > 1
        new_rear_neighbours = Array(nodes[1]:nodes[rear_index-1])
    else 
        new_rear_neighbours = []
    end

    if front_index[1] < length(nodes)
	new_front_neighbours = Array(nodes[front_index+1]:last(nodes))
    else
        new_front_neighbours = []
    end

    return new_rear_neighbours, new_front_neighbours, rear_index, front_index

end




    

    






@everywhere function equalities(sys::Int, x::Array, u::Array, parameters::Tuple)
    
    num_cars, T, N, _, _, (omega1, omega2), astart, _, asep, atot  = parameters

    constraints = Array{Any, 1}(undef, 2N+4)

    constraints[1:N] = [x[1, k+1] - x[1, k] - (T/N) * x[2, k] for k = 1:N]    # Displacement dynamics
    constraints[N+1:2N] = [x[2, k+1] - x[2, k] - (T/N) * u[k] for k = 1:N]    # Velocity dynamics

    constraints[2N+1] = x[1, 1] - sum(astart[1:sys])    # Initial displacements
    constraints[2N+2] = x[2, 1] - omega1[sys]           # Initial velocities

    constraints[2N+3] = x[1, N+1] - atot - sum(astart[1:num_cars]) + (num_cars - sys)*asep   # Final displacements
    constraints[2N+4] = x[2, N+1] - omega2[sys]                                              # Final velocities

    return constraints

end







@everywhere function uncoupled_inequalities(x::Array, u::Array, parameters::Tuple)

    _, _, N, (umin, umax), (omegaMin, omegaMax) = parameters

    constraints = Array{Any, 1}(undef, 4N+2)

    constraints[1:N] = [umin - u[k] for k = 1:N]       # Input lower limit
    constraints[N+1:2N] = [u[k] - umax for k = 1:N]    # Input upper limit

    constraints[2N+1:3N+1] = [omegaMin - x[2, k] for k = 1:N+1]    # Velocity lower limit
    constraints[3N+2:4N+2] = [x[2, k] - omegaMax for k = 1:N+1]    # Velocity upper limit

    return constraints

end








@everywhere function coupled_inequalities(x_behind::Array, x_ahead::Array, parameters::Tuple, is_lap=false)

    _, _, N, _, _, _, _, amin = parameters

    if is_lap
	return [amin - 2pi - (x_ahead[1, k] - x_behind[1, k]) for k = 1:N+1]
    else
	return [amin - (x_ahead[1, k] - x_behind[1, k]) for k = 1:N+1]
    end
 

end







@everywhere function base_model(sys::Int, parameters::Tuple, iter_limit::Int = 1000)

    _, _, N = parameters

    model = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model, "max_iter", iter_limit)

    x = Dict()    # State variables
    x[sys] = @variable(model, [1:2, 1:N+1], base_name = "x$sys")

    u = Dict()    # Input variables 
    u[sys] = @variable(model, [1:N], base_name = "u$sys")

    model[:x] = x    # Bind variables...
    model[:u] = u    # ... to model

    @objective(model, Min, sum(u[sys].^2))
    @constraint(model, equalities(sys, x[sys], u[sys], parameters) .== 0)          # Equality constraints
    @constraint(model, uncoupled_inequalities(x[sys], u[sys], parameters) .<= 0)   # Uncoupled inequality constraints
    
    return model

end







@everywhere function update_model(model::Model, nodes::Array, 
                                  prev_nodes::Array, parameters::Tuple)

    num_cars, _, N = parameters
    new_neighbours = setdiff(nodes, prev_nodes)
    new_rear_neighbours, new_front_neighbours, rear_index, front_index = 
        ordering_new_neighbourhood(nodes, prev_nodes, parameters)


    x = model[:x]
    u = model[:u]
    for j in new_neighbours
        x[j] = @variable(model, [1:2, 1:N+1], base_name = "x$j")
        u[j] = @variable(model, [1:N], base_name = "u$j")
    end
    

    @objective(model, Min, sum(sum([u[j].^2 for j in nodes])))
    @constraint(model, [j in new_neighbours], equalities(j, x[j], u[j], parameters) .== 0)
    @constraint(model, [j in new_neighbours], uncoupled_inequalities(x[j], u[j], parameters) .<= 0)


    if length(new_rear_neighbours) > 0
        @constraint(model, coupled_inequalities(x[last(new_rear_neighbours)], x[nodes[rear_index]], parameters,
                                                nodes[rear_index] == 1 ? true : false) .<= 0)
        if length(new_rear_neighbours) > 1
            for k = 1:length(new_rear_neighbours)-1
                @constraint(model, coupled_inequalities(x[new_rear_neighbours[k]], x[new_rear_neighbours[k+1]], parameters,
                                                        x[new_rear_neighbours[k]] == num_cars ? true : false) .<= 0)
            end
	end
    end


    if length(new_front_neighbours) > 0
        @constraint(model, coupled_inequalities(x[nodes[front_index]], x[new_front_neighbours[1]], parameters, 
                                                nodes[front_index] == num_cars ? true : false) .<= 0)
	if length(new_front_neighbours) > 1
            for k = 1:length(new_front_neighbours)-1
                @constraint(model, coupled_inequalities(x[new_front_neighbours[k]], x[new_front_neighbours[k+1]], parameters,
                                                        new_front_neighbours[k] == num_cars ? true : false) .<= 0)
            end
	end
    end
  
 
    nothing

end








@everywhere function optimise_model(model::Model)

    set_silent(model)
    optimize!(model)
    # println(termination_status(model))
    nothing
   
end

