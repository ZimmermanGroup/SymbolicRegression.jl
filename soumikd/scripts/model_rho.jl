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

using Plots

function mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
            return L(Inf)
    end

    mse_loss = sum((prediction .- dataset.y) .^ 2) / dataset.n

    return mse_loss

end

vxc_filename = "/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/vxc_smooth_output.txt"
exc_filename = "/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/EXCR"
rho_filename = "/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/n0"

keyword = "DFT quantities"

vxc_exc_rho_matrix = get_exc_vxc_rho(vxc_filename, exc_filename, rho_filename, keyword)

R = reshape(vxc_exc_rho_matrix[:, 1], :, 1)
rho = reshape(vxc_exc_rho_matrix[:, 2], :, 1)

# model = SRRegressor(
#     niterations= 1000,
#     populations= 10,
#     ncycles_per_iteration= 10,
#     binary_operators=(+, *, /, -,),
#     unary_operators=(exp,),
#     maxsize=10,
#     # constraints=(#=(^)=>(-1, 1),=#
#     #             (exp)=> 1,),
#     # procs=16,
#     parallelism=:multithreading,
#     # loss_function=my_custom_objective,
#     loss_function=mse_loss,
#     # output_directory="/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/exc_wc"
# )

# mach = machine(model, R, rho, scitype_check_level=0)

# fit!(mach)
# r = report(mach)
# r.equations[r.best_idx]

rho_predicted = 35.308 * exp.(-7.8514 .* R)
p = plot(R, [rho rho_predicted], label=["ρ" "35.308 * exp(-7.8514 r)"])
xlims!(0, 4)
xlabel!("r")
savefig(p, "rho_fit.png")