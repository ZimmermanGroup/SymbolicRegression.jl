using SymbolicRegression

defs = quote
    _plus(x, y) = x + y
    _mult(x, y) = x * y
    _div(x, y) = x / y
    _min(x, y) = x - y
    _cos(x) = cos(x)
    _exp(x) = exp(x)
    early_stop(loss, c) = ((loss <= 1e-10) && (c <= 10))
    my_loss(x, y, w) = abs(x - y)^2 * w
end

# This is needed as workers are initialized in `Core.Main`!
Core.eval(Core.Main, defs)
if (@__MODULE__) != Core.Main
    eval(:(using Main: _plus, _mult, _div, _min, _cos, _exp, early_stop, my_loss))
end

X = randn(Float32, 5, 100)
y = _mult.(2, _cos.(X[4, :])) + _mult.(X[1, :], X[1, :])

options = SymbolicRegression.Options(;
    binary_operators=(_plus, _mult, _div, _min),
    unary_operators=(_cos, _exp),
    populations=20,
    early_stop_condition=early_stop,
    elementwise_loss=my_loss,
)

hof = equation_search(
    X,
    y;
    weights=ones(Float32, 100),
    options=options,
    niterations=1_000_000_000,
    numprocs=2,
    parallelism=:multiprocessing,
)

@test any(
    early_stop(member.loss, count_nodes(member.tree)) for member in hof.members[hof.exists]
)
