using SymbolicRegression

include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/src/Utils.jl")
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/test/test_params.jl")
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/utils/process.jl")

import .UtilsModule: eval_limit, eval_derivative
using Symbolics
using SymbolicUtils
using DynamicDiff, DynamicExpressions

import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

using DelimitedFiles

function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}

    prediction, flag = eval_tree_array(tree, reshape(dataset.X[1,:], 1, :), options)
    if !flag
            return L(Inf)
    end

    prediction_loss = weighted_mse_loss(tree, dataset, options)

    exc_symbolic = node_to_symbolic(tree, options; variable_names=["x1"], index_functions=true)

    if (occursin("x1", repr(tree)))

        if (occursin("x2", repr(tree)) || occursin("x3", repr(tree)) || occursin("x4", repr(tree)))
            return prediction_loss + 10000000
        end

        # check the limit of exc at infinity to be -1/2r
        @syms x1
        modified_exc_tree = symbolic_to_node(x1*exc_symbolic, options;  variable_names=["x1"])

        lim1 = eval_limit(modified_exc_tree, 1e10, options)
        lim_loss1 = abs(lim1 + 0.5)

        # check the functional derivative of exc almost matches with vxc

        r = reshape(dataset.X[1,:], :, 1)
        rho = reshape(dataset.X[2,:], :, 1)
        drho_dr = reshape(dataset.X[3,:], :, 1)
        vxc = reshape(dataset.X[end,:], :, 1)

        dexc_dr = eval_derivative(tree, r, options)
        exc_val, flag = eval_tree_array(tree, reshape(r, 1, :), options)
        # Added a padding term to drho_dr so that it doesn't vanish
        weighted_lim_loss2 = sum(dataset.weights .* (vxc .- exc_val .- (rho .*dexc_dr ./ (drho_dr .+ 1e-8))).^2)/(dataset.n * sum(dataset.weights))

        # check the numerical value of vxc at large distance to be -1/r

        vxc_pred = exc_val[end][1] + rho[end][1]*dexc_dr[end][1]/(drho_dr[end][1] + 1e-8)
        lim_loss3 = abs(r[end][1]*vxc_pred + 1)

        return 100*prediction_loss + lim_loss1 + 100*weighted_lim_loss2 + 10*lim_loss3

    else
        return prediction_loss + 10000000
    end

end

function weighted_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
      
    prediction, flag = eval_tree_array(tree, reshape(dataset.X[1,:], 1, :), options)
    if !flag
            return L(Inf)
    end

    weighted_mse_loss = sum(dataset.weights.* (prediction .- dataset.y) .^ 2) / (dataset.n * sum(dataset.weights))

    return weighted_mse_loss

end


vxc_filename = "/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/vxc_smooth_output.txt"
exc_filename = "/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/EXCR"
rho_filename = "/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/n0"

keyword = "DFT quantities"

vxc_exc_rho_matrix = get_exc_vxc_rho(vxc_filename, exc_filename, rho_filename, keyword)

R = reshape(vxc_exc_rho_matrix[:, 1], :, 1)
exc = reshape(vxc_exc_rho_matrix[:, end], :, 1)
vxc = reshape(vxc_exc_rho_matrix[:, end-1], :, 1)
rho = reshape(vxc_exc_rho_matrix[:, 2], :, 1)
drho_dr = reshape((rho[2:end] .- rho[1:end-1]) ./ (R[2:end] .- R[1:end-1]), :, 1)

idx = findfirst(item -> item < 1e-7, rho)[1] - 1

epsilon_xc = reshape(exc[1:idx] ./ rho[1:idx], :, 1)
truncated_R = reshape(R[1:idx], :, 1)
weights = reshape(truncated_R.^2 .* rho[1:idx], :, 1)

# epsilon_xc = reshape(exc ./ rho, :, 1)
# weights = reshape(R.^2 .* rho, :, 1)

println("Data loaded !! Now loading SRRegressor object")

model = SRRegressor(
    niterations= 1000,
    populations= 10,
    ncycles_per_iteration= 10,
    binary_operators=(+, *, /, -,),
    unary_operators=(exp,),
    maxsize=20,
    constraints=(#=(^)=>(-1, 1),=#
                (exp)=> 1,),
    # procs=16,
    parallelism=:multithreading,
    loss_function=my_custom_objective,
    # loss_function=weighted_mse_loss,
    output_directory="/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/exc_wc"
)

mach = machine(model, [truncated_R rho[1:idx] drho_dr[1:idx] vxc[1:idx]], epsilon_xc, weights, scitype_check_level=0)

# mach = machine(model, truncated_R, epsilon_xc, weights, scitype_check_level=0)

println("Starting the fitting process !!")

fit!(mach)
r = report(mach)
r.equations[r.best_idx]
# predict(mach, X)