using Profile
using SymbolicUtils
using SymbolicRegression

using DynamicExpressions: string_tree
using Symbolics

import SymbolicRegression: SRRegressor, UtilsModule
# import .UtilsModule: eval_limit, eval_derivative, fnFromString
import MLJ: machine, fit!, predict, report

function fnFromString(s)
    f = eval(Meta.parse("x -> " * s))
    return x -> Base.invokelatest(f, x)
end

function eval_limit(f::Function, val::Float64)
    @syms x
    var = "x"

    # if (occursin(var, eq_str))
    try
        f_val = f(val)

        if (f_val == NaN)
            fn_value_x = f(x)
            return limit(fn_value_x, x, val)
        else
            return f_val
        end
    catch e
        if e isa DomainError
            return 10000  # an arbitrary large number
        else
            rethrow(e)
        end
    end
    # else
    #     return 10000  # an arbitrary large number
    # end
end

function get_deriv_from_fn(f::Function)
    @syms x

    # g = fnFromString(f)

    deriv = expand_derivatives(Differential(x)(eval(f)(x)))
    return build_function(deriv, x; expression=Val{false})
end

function eval_derivative(f::Function, val::Float64)
    @syms x

    # if (occursin(var, eq_str))

    # new_eq_str = replace(eq_str, var => "x")

    # if(eq_str == "x")
    #     return NaN, 1
    # end

    # f = fnFromString(new_eq_str)

    try
        df = get_deriv_from_fn(f)

        if (occursin("x", repr(df)))
            df_val = df(val)
            df_x = df(x)
            if (df_val == NaN)
                return repr(df_x), limit(df_x, x, val)
            else
                return repr(df_x), df_val
            end
        else
            try
                return NaN, parse(Float64, @show simplify(df))
            catch e
                if e isa ArgumentError
                    return NaN, 10000
                else
                    rethrow(e)
                end
            end
        end
    catch e
        if e isa DomainError
            return NaN, 10000
        else
            rethrow(e)
        end
    end

    # else
    #     return NaN, 10000  # a random large number
    # end
end

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

    eq_str = replace(eq_str, vrble => "x")
    vrble = "x"

    prediction_loss = mse_loss(tree, dataset, options)

    lambda1 = 1000
    lambda2 = 1000
    lambda3 = 1000
    lambda4 = 1000

    if (occursin(vrble, eq_str))

        f = fnFromString(eq_str)

        lim1 = eval_limit(f, 0.0)
        lim_loss1 = abs(1 - lim1)
        
        f2, lim2 = eval_derivative(f, 0.0)
        lim_loss2 = abs(1 - lim2)

        # if (f2 != NaN)
        #     f3, lim3 = eval_derivative(repr(f2), vrble, 0.0)
        #     lim_loss3 = abs(1 + lim3)

        #     if (f3 != NaN)
        #         f4, lim4 = eval_derivative(repr(f3), vrble, 0.0)
        #         lim_loss4 = abs(2 + lim4)

        #         return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2 + lambda3*lim_loss3 + lambda4*lim_loss4

        #     else
        #         return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2 + lambda3*lim_loss3
        #     end

        # else
        #     return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2
        # end
        return prediction_loss + lambda1*lim_loss1 + lambda2*lim_loss2
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



# model = SRRegressor(
#     niterations= 1,
#     populations= 2,
#     # population_size= 3,
#     ncycles_per_iteration= 1,
#     # binary_operators=[+, *, /, -],
#     binary_operators=[+, *, -],
#     # unary_operators=[],
#     maxsize=8,
#     # procs=16,
#     parallelism=:multithreading,
#     loss_function=my_custom_objective,
#     # loss_function=mse_loss,
# )

# mach = machine(model, X, f, scitype_check_level=0)
# Profile.init(delay=0.1)
# # using ProfileView
# # ProfileView.view()
# @profile fit!(mach)
# open("profile_output.txt", "w") do f
#     Profile.print(IOContext(f, :displaysize => (24, 500)), mincount=100, maxdepth=100)
# end
# Profile.clear()
# report(mach)
# predict(mach, X)

using Symbolics
include("/home/joshkamm/SymbolicRegression/SymbolicRegression.jl/test/test_params.jl")

_inv(x) = 1 / x
options = Options(;
default_params...,
binary_operators=(+, *, ^, /),
unary_operators=(_inv,),
constraints=(_inv => 4,),
populations=4,
)
@extend_operators options
tree = Node(1, (^)(Node(; val=3.0) * Node(1, Node("x1")), 2.0), Node(; val=-1.2))

node_to_symbolic(tree, options; variable_names=["energy"], index_functions=true)

dataset = Dataset(randn(3, 32), randn(Float32, 32); weights=randn(Float32, 32))

@time mse_loss(tree, dataset, options)
@time mse_loss(tree, dataset, options)
@time my_custom_objective(tree, dataset, options)
@time my_custom_objective(tree, dataset, options)
