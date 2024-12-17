using SymbolicRegression

using DynamicExpressions: string_tree
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/src/Utils.jl")
import .UtilsModule: eval_limit, eval_derivative
using Symbolics

@time function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
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

@time function mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
            return L(Inf)
    end

    mse_loss = sum((prediction .- dataset.y) .^ 2) / dataset.n

    return mse_loss

end