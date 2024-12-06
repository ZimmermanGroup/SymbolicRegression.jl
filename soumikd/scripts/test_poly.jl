using Profile
using SymbolicRegression

using DynamicExpressions: string_tree
using Symbolics

import SymbolicRegression: SRRegressor, UtilsModule
import .UtilsModule: eval_limit, eval_derivative
import MLJ: machine, fit!, predict, report

function my_custom_profiling_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # @profile mse_loss(tree, dataset, options)
    @profile my_custom_objective(tree, dataset, options)
end

function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
            return L(Inf)
    end
    
    eq_str = string_tree(tree, options)
    # print(eq_str)

    vrble = "x1"

    prediction_loss = mse_loss(tree, dataset, options)

    lambda1 = 1000
    lambda2 = 1000
    lambda3 = 1000
    lambda4 = 1000

    if (occursin(vrble, eq_str))

        lim1 = eval_limit(eq_str, vrble, 0.0)
        lim_loss1 = abs(1 - lim1)
        
        f2, lim2 = eval_derivative(eq_str, vrble, 0.0)
        lim_loss2 = abs(1 - lim2)

        if (f2 != NaN)
            f3, lim3 = eval_derivative(repr(f2), vrble, 0.0)
            lim_loss3 = abs(1 + lim3)

            if (f3 != NaN)
                f4, lim4 = eval_derivative(repr(f3), vrble, 0.0)
                lim_loss4 = abs(2 + lim4)

                return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2 + lambda3*lim_loss3 + lambda4*lim_loss4

            else
                return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2 + lambda3*lim_loss3
            end

        else
            return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2
        end
        # return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2
    else
        return prediction_loss + 10000
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
X = reshape(X, 11,1)
f = -X.^3/3 - X.^2/2 + X .+ 1

model = SRRegressor(
    niterations= 1,
    populations= 2,
    # population_size= 3,
    ncycles_per_iteration= 1,
    # binary_operators=[+, *, /, -],
    binary_operators=[+, *, -],
    # unary_operators=[],
    maxsize=8,
    # procs=16,
    parallelism=:multithreading,
    loss_function=my_custom_objective,
    # loss_function=mse_loss,
)

mach = machine(model, X, f, scitype_check_level=0)
Profile.init(delay=0.1)
# using ProfileView
# ProfileView.view()
@profile fit!(mach)
open("profile_output.txt", "w") do f
    Profile.print(IOContext(f, :displaysize => (24, 500)), mincount=100)
    # Profile.print(IOContext(f, :displaysize => (24, 500)), mincount=10, maxdepth=10)
end
Profile.clear()
report(mach)
predict(mach, X)
