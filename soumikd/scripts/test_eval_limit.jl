using SymbolicRegression
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/src/Utils.jl")
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/test/test_params.jl")
import .UtilsModule: eval_limit
using Symbolics
using SymbolicUtils

import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
            return L(Inf)
    end

    prediction_loss = mse_loss(tree, dataset, options)

    f = node_to_symbolic(tree, options; variable_names=["x1"], index_functions=true)

    if (occursin("x1", repr(f)))

        lim1 = eval_limit(f, 0.0, options)
        lim_loss1 = abs(1 - lim1)

        return prediction_loss + lim_loss1
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