@testitem "pretty print member" tags = [:part3] begin
    using SymbolicRegression

    options = Options(; binary_operators=[+, ^])

    ex = @parse_expression(x^2.0 + 1.5, binary_operators = [+, ^], variable_names = [:x])
    shower(x) = sprint((io, e) -> show(io, MIME"text/plain"(), e), x)
    s = shower(ex)
    @test s == "(x ^ 2.0) + 1.5"

    X = [1.0 2.0 3.0]
    y = [2.0, 3.0, 4.0]
    dataset = Dataset(X, y)
    member = PopMember(dataset, ex, options; deterministic=false)
    member.score = 1.0
    @test member isa PopMember{Float64,Float64,<:Expression{Float64,Node{Float64}}}
    s_member = shower(member)
    @test s_member == "PopMember(tree = ((x ^ 2.0) + 1.5), loss = 16.25, score = 1.0)"

    # New options shouldn't change this
    options = Options(; binary_operators=[-, /])
    s_member = shower(member)
    @test s_member == "PopMember(tree = ((x ^ 2.0) + 1.5), loss = 16.25, score = 1.0)"
end

@testitem "pretty print hall of fame" tags = [:part1] begin
    using SymbolicRegression
    using SymbolicRegression: embed_metadata
    using SymbolicRegression.CoreModule: safe_pow

    options = Options(; binary_operators=[+, safe_pow], maxsize=7)

    ex = @parse_expression(
        $safe_pow(x, 2.0) + 1.5, binary_operators = [+, safe_pow], variable_names = [:x]
    )
    shower(x) = sprint((io, e) -> show(io, MIME"text/plain"(), e), x)
    s = shower(ex)
    @test s == "(x ^ 2.0) + 1.5"

    X = [1.0 2.0 3.0]
    y = [2.0, 3.0, 4.0]
    dataset = Dataset(X, y)
    member = PopMember(dataset, ex, options; deterministic=false)
    member.score = 1.0
    @test member isa PopMember{Float64,Float64,<:Expression{Float64,Node{Float64}}}

    hof = HallOfFame(options, dataset)
    hof = embed_metadata(hof, options, dataset)
    hof.members[5] = member
    hof.exists[5] = true
    s_hof = strip(shower(hof))
    true_s = "HallOfFame{...}:
    .exists[1] = false
    .members[1] = undef
    .exists[2] = false
    .members[2] = undef
    .exists[3] = false
    .members[3] = undef
    .exists[4] = false
    .members[4] = undef
    .exists[5] = true
    .members[5] = PopMember(tree = ((x ^ 2.0) + 1.5), loss = 16.25, score = 1.0)
    .exists[6] = false
    .members[6] = undef
    .exists[7] = false
    .members[7] = undef"

    @test s_hof == true_s
end

@testitem "pretty print expression" tags = [:part2] begin
    using SymbolicRegression
    using Suppressor: @capture_out

    options = Options(; binary_operators=[+, -, *, /], unary_operators=[cos])
    ex = @parse_expression(
        cos(x) + y * y, operators = options.operators, variable_names = [:x, :y]
    )

    s = sprint((io, ex) -> print_tree(io, ex, options), ex)
    @test strip(s) == "cos(x) + (y * y)"

    s = @capture_out begin
        print_tree(ex, options)
    end
    @test strip(s) == "cos(x) + (y * y)"

    # Works with the tree itself too
    s = @capture_out begin
        print_tree(get_tree(ex), options)
    end
    @test strip(s) == "cos(x1) + (x2 * x2)"
    s = sprint((io, ex) -> print_tree(io, ex, options), get_tree(ex))
    @test strip(s) == "cos(x1) + (x2 * x2)"

    # Updating options won't change printout, UNLESS
    # we pass the options.
    options = Options(; binary_operators=[/, *, -, +], unary_operators=[sin])

    s = @capture_out begin
        print_tree(ex)
    end
    @test strip(s) == "cos(x) + (y * y)"

    s = sprint((io, ex) -> print_tree(io, ex, options), ex)
    @test strip(s) == "sin(x) / (y - y)"
end

@testitem "printing utilities" tags = [:part2] begin
    using SymbolicRegression.UtilsModule: split_string
    using SymbolicRegression.HallOfFameModule: wrap_equation_string

    @test split_string("abc\ndefg", 3) == ["abc", "\nde", "fg"]
    @test wrap_equation_string("abcdefghijklmnop\nqrs\ntuvwxyz", 10, 30) ==
        "abcdefghijklmnop...\n          qrs...\n          tuvwxyz\n"
end
