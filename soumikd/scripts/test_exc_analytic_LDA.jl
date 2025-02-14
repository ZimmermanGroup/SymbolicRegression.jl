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
using Plots


function my_custom_objective(tree, dataset::Dataset{T,L}, options)::L where {T,L}

    # rho is modelled as a exp(-gr) + br^2 exp(-hr) #
    a = 33.82613608
    b = 0.19754759
    g = 7.58284232
    h = 1.95816344

    prediction, flag = eval_tree_array(tree, reshape(dataset.X[1,:], 1, :), options)
    if !flag
            return L(Inf)
    end

    prediction_loss = weighted_mse_loss(tree, dataset, options)

    exc_symbolic = node_to_symbolic(tree, options; variable_names=["x1"], index_functions=true)

    if (occursin("x1", repr(tree)))

        if (occursin("x2", repr(tree)))
            return prediction_loss + 10000000
        end

        # check the limit of exc at infinity to be -1/2r
        @syms x1
        modified_exc_tree = symbolic_to_node(x1*exc_symbolic, options;  variable_names=["x1"])

        lim1 = eval_limit(modified_exc_tree, 1e10, options)
        lim_loss1 = 0
        if (lim1 == Inf)
            lim_loss1 = 1e6
        else
            lim_loss1 = abs(lim1 + 0.5)
        end

        # check the functional derivative of exc almost matches with vxc

        r = reshape(dataset.X[1,:], :, 1)
        vxc = reshape(dataset.X[end,:], :, 1)

        dexc_dr = eval_derivative(tree, r, options)
        exc_val, flag = eval_tree_array(tree, reshape(r, 1, :), options)

        rho_symbolic = (a .* exp.(- g .* r)) .+ (b .* r.^2 .* exp.(- h .* r))
        drho_dr_symbolic = (- a*g .* exp.(- g .* r)) + ((2*b .* r - b*h .* r.^2).* exp.(- h .* r))

        weighted_lim_loss2 = sum(dataset.weights .* (vxc .- exc_val .- ((dexc_dr .* rho_symbolic)./drho_dr_symbolic)).^2)/(dataset.n * sum(dataset.weights))

        # check the numerical value of vxc at large distance to be -1/r
        # from the LDA approximation at long range vxc ~ exc - 1/h dexc/dr
        # In other words, dexc/dr has to vary as h/2r

        dexc_dr_at_infinity = 1e6 * eval_derivative(tree, [1e6], options)[1][1] / h
 
        lim_loss3 = 0
        if (dexc_dr_at_infinity != NaN)
            lim_loss3 = abs(dexc_dr_at_infinity - 0.5)
        else
            lim_loss3 = 1e6
        end

        # return 10*prediction_loss + lim_loss1 + 100*weighted_lim_loss2 + 10*lim_loss3
        return 10*prediction_loss + lim_loss1 + 100*weighted_lim_loss2

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
rho = reshape(vxc_exc_rho_matrix[:, 2], :, 1)
exc = reshape(vxc_exc_rho_matrix[:, end], :, 1)
vxc = reshape(vxc_exc_rho_matrix[:, end-1], :, 1)

idx = findfirst(item -> item < 1e-7, rho)[1] - 1

epsilon_xc = reshape(exc[1:idx] ./ rho[1:idx], :, 1)
weights = reshape(R[1:idx].^2 .* rho[1:idx], :, 1)
# weights = reshape(rho[1:idx], :, 1)

println("Data loaded !! Now loading SRRegressor object")

model = SRRegressor(
    niterations= 1000,
    populations= 10,
    ncycles_per_iteration= 10,
    binary_operators=(+, *, /, -,),
    unary_operators=(exp,),
    maxsize=20,
    constraints=((^)=>(-1, 1),
                (exp)=> 1,),
    # procs=16,
    parallelism=:multithreading,
    loss_function=my_custom_objective,
    # loss_function=weighted_mse_loss,
    output_directory="/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/exc_wc"
)

mach = machine(model, [R[1:idx] vxc[1:idx]], epsilon_xc, weights, scitype_check_level=0)

# mach = machine(model, truncated_R, epsilon_xc, weights, scitype_check_level=0)

println("Starting the fitting process !!")

fit!(mach)
r = report(mach)
r.equations[r.best_idx]
# predict(mach, X)