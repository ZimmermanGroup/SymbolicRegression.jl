using SymbolicRegression
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/src/Utils.jl")
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/test/test_params.jl")
using Symbolics
using SymbolicUtils
using DynamicDiff, DynamicExpressions

import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

function eval_derivative(tree, val, options)
    try
        operators = OperatorEnum(; binary_operators=options.operators.binops, 
                                       unary_operators=options.operators.unaops)
        variable_names = ["x1"]
        x1 = (Expression(Node{Float64}(feature=1); operators, variable_names))

        f = Expression(tree; operators, variable_names)
        df_val = D(f, 1)(val')

        if (NaN in df_val)
            return 10000
        else
            return df_val
        end
    catch e
        if e isa DomainError
            return 10000  # a random large number
        else
            rethrow(e)
        end
    end
end

function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
            return L(Inf)
    end

    prediction_loss = mse_loss(tree, dataset, options)

    if (occursin("x1", repr(tree)))

        lim1 = eval_derivative(tree, [0.0, 1.0], options)
        lim_loss1 = abs(sum([1, -1] .- lim1))

        return prediction_loss + 10*lim_loss1
    else
        return prediction_loss + 100000
    end

end

function mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
            return L(Inf)
    end

    mse_loss = sum((prediction .- dataset.y) .^ 2) / dataset.n

    return mse_loss

end

X = [100:110;]
X = Array{Float64}(X)
X = reshape(X, 11, 1)
f = -X.^3/3 - X.^2/2 + X .+ 1

model = SRRegressor(
    niterations= 100,
    populations= 10,
    ncycles_per_iteration= 10,
    binary_operators=(+, *, /, -,),
    # unary_operators=[],
    maxsize=20,
    # procs=16,
    parallelism=:multithreading,
    loss_function=my_custom_objective,
)

mach = machine(model, X, f, scitype_check_level=0)
fit!(mach)
report(mach)
predict(mach, X)