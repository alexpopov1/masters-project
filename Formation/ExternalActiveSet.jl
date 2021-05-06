# This code implements the external active set strategy, as defined in Chung
# et al (2010). Paper can be found here:
# https://www.researchgate.net/publication/225394821_On_the_Use_of_Outer_Approximations_as_an_External_Active_Set_Strategy
#The algorithm in the paper as been amended to work with two other possible
# heuristics (% and the combined ϵ-%), and the code allows the user to choose
# which strategy to implement.



# For optimisation solver
using JuMP, Ipopt

# Include additional function files
include("WarmStart.jl")







## Define required functions for algorithm




# Maximum constraint function value for a given trajectory set and vector of
# constraint functions

function max_constraint(states, con_function)

        return maximum(con_function(states))

end







# Enforce a non-negative condition on maxConstraint(), so that the function
# returns zero if all constraints are satisfied

function max_pos_constraint(states, con_function)

        return max(max_constraint(states, con_function), 0)

end







# Define a threshold function, setting the  range of constraint function values
# which are considered close enough to maximum to be included in ϵ-active set

function threshold_indicator(states, con_function, threshold)

        return min(max_pos_constraint(states, con_function), threshold)

end








# Construct the index set (pointing to expected most volatile constraints) for a
# given state space trajectory, containing constraints within a threshold
# distance of the maximum constraint value

function make_threshold_set(states, con_function, threshold)

        indexSet = Float64[]
        constraint_eval_set = cons_function(states)
        max_val = max_pos_constraint(states, con_function)
        set_range = threshold_indicator(states, con_function, threshold)
        threshold_set = findall(i -> i >= max_val - set_range, constraint_eval_set)

        return threshold_set

end









# Returns the indices of all violated constraints, as well as the values of the
# corresponding constraints

function constraint_violations(states, con_function)

        indices = findall(i -> i > 0, con_function(states))
        vals = [con_function(states)[i] for i in indices]
        return indices, vals

end









# Construct an index set pointing to a given percentage of all violated NL
# constraints, starting from the maximum value i.e. if percentage = 10,
# the set will contain the indices pointing to the largest 10% of the violated
# constraints. This function also returns the values of all violated constraints.

function make_percent_set(states, con_function, percentage)

        indices, vals = constraint_violations(states, con_function)
        viol_constraint_vals = vals
        limit = Int(ceil(0.01 * percentage * length(indices)))
        percent_set = Array{Int64}(undef, limit)

        for i = 1:limit
            max_finder = findmax(vals)[2]
            percent_set[i] = indices[max_finder]
            vals[max_finder] = 0
        end

        return percent_set, viol_constraint_vals

end







# Constructs the index set of constraints to be added to the external active
# set at a given iteration of the algorithm. This function makes use of
# the percentage and threshold set constructors, combined with the conditional
# statements to ensure that the index set can be constructed according to
# flexible user inputs.
function make_index_set(states, con_function, threshold, percentage, tracker)

        if tracker || percentage > 0
            percent_set, viol_set = make_percent_set(states, con_function, percentage)
        else
            percent_set = viol_set = Array{Int64}(undef, 0)
        end


        if threshold != Inf
            threshold_set = make_threshold_set(states, con_function, threshold)
        else
            threshold_set = Array{Int64}(undef, 0)
        end


        # The index set is defined as the larger of the two sets
        if length(percent_set) > length(threshold_set)
            index_set = percent_set
        else
            index_set = threshold_set
        end


        return index_set, viol_set

end







## External active set algorithm



# Arguments:

#    model        - A JuMP model, fully constructed except for the nonlinear
#                   inequality constraints, which will be defined separately.

#    variables    - A tuple (x, u) where x and u are JuMP variables
#                   representing the states and inputs respectively.

#    start        - A tuple (x₀, u₀) which contains initial values for the
#                   states and inputs, to serve as a warm start for the
#                   initial optimisation problem.

#    constraints  - A generic function taking the states as an argument and
#                   returns a vector of functions for all NL inequality
#                   constraints in the problem.

#    threshold    - A scalar value which dictates the range of values from the
#                   maximum (ψ) that should be included in the ϵ-active set.
#                   This is equivalent to the constant term in the ϵ function.
#                   The default value is set to Inf, equivalent to including all
#                   violated constraints in the external active set.

#    percentage   - A scalar value indicating the percentage of all violated
#                   constraints which should be added the external active
#                   set. In combination with a non-trivial threshold value, this
#                   argument sets the minimum percentage of violated constraints
#                   to be added to the set.

#    N_iter       - The maximum number of iterations for which to run the
#                   optimisation solver.

#    track_cons   - Boolean variable stating whether violated constraint values
#                   should be computed and stored at each outer iteration. This
#                   This is only useful in the case where the threshold value
#                   is being used without any percentage cutoff, as the
#                   constraint values are not required for the algorithm itself.
#                   The user may still want to track these values for plots,
#                   but if not, then the values should not be computed as it
#                   will significantly increase the runtime.


# Outputs:

#    trajectory  - The optimal trajectory, as determined by the algorithm.

#    control     - Corresponding control inputs for solution

#    Q           - The set of indices pointing to functions which have appeared
#                  in the ϵ-active set at least once throughout the course of
#                  the algorithm.

#   iterations   - The number of iterations before algorithm terminates.

#   violated_cons - Array constaining the violated NL inequality constraints at
#                  each iteration of the algorithm.


function external_active_set(model; variables, constraints, start,
                           N_iter = 30, threshold = Inf, percentage = 0,
                           track_cons = false)

        # Set up
        trajectory, input_sequence = start
        x, u = variables
        set_optimizer_attribute(model, "max_iter", N_iter)

        # Prepare all possible active constraints
        all_constraints = constraints(x)

        # Initialise iterating index
        i = 0

        # Initialise remaining variables
        Q_prev = Int64[]
        violated_cons = Array{Array{Float64}}(undef, 0)

        # Construct index set (and set of all violated constraints, if needed)
        index_set, viol_set = make_index_set(trajectory, constraints, threshold,
                                          percentage, track_cons)

        # Initialise Q with first index set
        Q = indexSet

        # Infinite while loop to run algorithm iteration
        while true

                # Print information from current iteration
                println("\n", "i = ", i)
                println("Length of Q: ", length(Q))
                println("Maximum constraint value = ", max_constraint(trajectory, constraints))

                # Record current number of violated constraints
                push!(violated_cons, viol_set)

                # Warm start for current iteration
                warm_start(x, u, trajectory, input_sequence)

                # Add latest set of ϵ-active constraints to JuMP model
                for j in setdiff(Q, Q_prev)
                    @constraint(model, all_constraints[j] <= 0)
                end

                # Run solver
                optimize!(model)

                # Set trajectory and input sequence with optimisation solutions
                trajectory = value.(x)
                inputSequence = value.(u)

                # Check for algorithm completion - instead of strictly requiring
                # an optimal solution, an alternative condition of a stable Q
                # avoids the case of the algorithm never terminating even though
                # all constraints have been satisfied and Q is not changing.

                if max_constraint(trajectory, constraints) <= 1e-5 &&
                   (termination_status(model) == MOI.LOCALLY_SOLVED ||
                   length(Q) == length(Q_prev))
                    break
                end

                # Store current Q before it is updated
                Q_prev = Q

                # Construct index set (and set of all violated constraints, if needed)
                index_set, viol_set = make_index_set(trajectory, constraints, threshold,
                                                  percentage, track_cons)

                # Update Q to include any new constraints from the current iteration
                Q = union(Q, index_set)

                # Update iterating index
                i = i + 1

        end
        
        control = input_sequence

        # Record violated constraints at termination (trivial - will be empty)
        push!(violated_cons, Array{Float64}(undef,0))

        # Set number of iterations completed by the algorithm
        i_terminate = i + 1

        # Print algorithm properties
        println("\n", "Iterations of algorithm: ", i_terminate)
        println("Number of functions in Q: ", length(Q))


        # Function outputs
        return trajectory, control, Q, i_terminate, violated_cons

end



##
