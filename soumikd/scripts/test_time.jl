using SymbolicRegression

include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/src/Utils.jl")
include("/home/soumikd/symbolic_regression/SymbolicRegression.jl/test/test_params.jl")

using Symbolics
using SymbolicUtils
using DynamicDiff
using DynamicExpressions: OperatorEnum

_inv(x) = 1 / x
options = Options(;
default_params...,
binary_operators=(+, *, ^, /),
unary_operators=(_inv,),
constraints=(_inv => 4,),
populations=4,
)
println(options.operators.unaops)
println(options.operators.binops)

@extend_operators options
tree = Node(1, (^)(Node(; val=3.0) * Node(1, Node("x1")), 2.0), Node(; val=-1.2))
tree2 = Node(1, (^)(Node(; val=5.0) * Node(1, Node("x1")), 2.0), Node(; val=-1.2))

@time f = node_to_symbolic(tree, options; variable_names=["x1"], index_functions=true)
@time g = node_to_symbolic(tree2, options; variable_names=["x1"], index_functions=true)

# @syms x1 y t
# @time g = build_function(f, x1)
# @time eval(g)(1)
# @time eval(g)(1)
# println(eval(g)(1))

# h(t) = sin(t)/t
# @time limit(h, t, 0)
# @time g(1)
# @time occursin("x1", repr(f))

@time eval_tree_array(tree, reshape(Array{Float64}([1.0]), 1, 1), options)
@time eval_tree_array(tree2, reshape(Array{Float64}([1.0]), 1, 1), options)
# println(value)

@time h = symbolic_to_node(f, options; variable_names=["x1"])
@time p = symbolic_to_node(g, options; variable_names=["x1"])


@time eval_tree_array(h, reshape(Array{Float32}([1.0]), 1, 1), options)
@time eval_tree_array(p, reshape(Array{Float32}([1.0]), 1, 1), options)[1]

operators = OperatorEnum(; binary_operators=(+, *, /, -, ^), unary_operators=(sin, cos));
variable_names = ["x1"];
x1 = (Expression(Node{Float64}(feature=1); operators, variable_names))

q = Expression(tree2; operators, variable_names)

# println(typeof(x1))
# println(typeof(q))
# println(typeof(2^x1))

@time D(q, 1)([1]')[1][1]