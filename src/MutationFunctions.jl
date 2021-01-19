# Randomly convert an operator into another one (binary->binary;
# unary->unary)
function mutateOperator(tree::Node, options::Options)::Node
    if countOperators(tree) == 0
        return tree
    end
    node = randomNode(tree)
    while node.degree == 0
        node = randomNode(tree)
    end
    if node.degree == 1
        node.op = rand(1:options.nuna)
    else
        node.op = rand(1:options.nbin)
    end
    return tree
end

# Randomly perturb a constant
function mutateConstant(
        tree::Node, T::Float32,
        options::Options)::Node
    # T is between 0 and 1.

    if countConstants(tree) == 0
        return tree
    end
    node = randomNode(tree)
    while node.degree != 0 || node.constant == false
        node = randomNode(tree)
    end

    bottom = 0.1f0
    maxChange = options.perturbationFactor * T + 1.0f0 + bottom
    factor = maxChange^Float32(rand())
    makeConstBigger = rand() > 0.5

    if makeConstBigger
        node.val *= factor
    else
        node.val /= factor
    end

    if rand() > options.probNegate
        node.val *= -1
    end

    return tree
end

# Add a random unary/binary operation to the end of a tree
function appendRandomOp(tree::Node, options::Options, nfeatures::Int)::Node
    node = randomNode(tree)
    while node.degree != 0
        node = randomNode(tree)
    end

    choice = rand()
    makeNewBinOp = choice < options.nbin/(options.nuna + options.nbin)
    if rand() > 0.5
        left = Float32(randn())
    else
        left = rand(1:nfeatures)
    end
    if rand() > 0.5
        right = Float32(randn())
    else
        right = rand(1:nfeatures)
    end

    if makeNewBinOp
        newnode = Node(
            rand(1:options.nbin),
            left,
            right
        )
    else
        newnode = Node(
            rand(1:options.nuna),
            left
        )
    end
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return tree
end

# Insert random node
function insertRandomOp(tree::Node, options::Options, nfeatures::Int)::Node
    node = randomNode(tree)
    choice = rand()
    makeNewBinOp = choice < options.nbin/(options.nuna + options.nbin)
    left = copyNode(node)

    if makeNewBinOp
        right = randomConstantNode(nfeatures)
        newnode = Node(
            rand(1:options.nbin),
            left,
            right
        )
    else
        newnode = Node(
            rand(1:options.nuna),
            left
        )
    end
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return tree
end

# Add random node to the top of a tree
function prependRandomOp(tree::Node, options::Options, nfeatures::Int)::Node
    node = tree
    choice = rand()
    makeNewBinOp = choice < options.nbin/(options.nuna + options.nbin)
    left = copyNode(tree)

    if makeNewBinOp
        right = randomConstantNode(nfeatures)
        newnode = Node(
            rand(1:options.nbin),
            left,
            right
        )
    else
        newnode = Node(
            rand(1:options.nuna),
            left
        )
    end
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return node
end

function randomConstantNode(nfeatures::Int)::Node
    if rand() > 0.5
        val = Float32(randn())
    else
        val = rand(1:nfeatures)
    end
    newnode = Node(val)
    return newnode
end


# Return a random node from the tree with parent
function randomNodeAndParent(tree::Node, parent::Union{Node, Nothing})::Tuple{Node, Union{Node, Nothing}}
    if tree.degree == 0
        return tree, parent
    end
    a = countNodes(tree)
    b = 0
    c = 0
    if tree.degree >= 1
        b = countNodes(tree.l)
    end
    if tree.degree == 2
        c = countNodes(tree.r)
    end

    i = rand(1:1+b+c)
    if i <= b
        return randomNodeAndParent(tree.l, tree)
    elseif i == b + 1
        return tree, parent
    end

    return randomNodeAndParent(tree.r, tree)
end

# Select a random node, and replace it an the subtree
# with a variable or constant
function deleteRandomOp(tree::Node, options::Options, nfeatures::Int)::Node
    node, parent = randomNodeAndParent(tree, nothing)
    isroot = (parent === nothing)

    if node.degree == 0
        # Replace with new constant
        newnode = randomConstantNode(nfeatures)
        node.l = newnode.l
        node.r = newnode.r
        node.op = newnode.op
        node.degree = newnode.degree
        node.val = newnode.val
        node.constant = newnode.constant
    elseif node.degree == 1
        # Join one of the children with the parent
        if isroot
            return node.l
        elseif parent.l == node
            parent.l = node.l
        else
            parent.r = node.l
        end
    else
        # Join one of the children with the parent
        if rand() < 0.5
            if isroot
                return node.l
            elseif parent.l == node
                parent.l = node.l
            else
                parent.r = node.l
            end
        else
            if isroot
                return node.r
            elseif parent.l == node
                parent.l = node.r
            else
                parent.r = node.r
            end
        end
    end
    return tree
end


# Create a random equation by appending random operators
function genRandomTree(length::Integer, options::Options, nfeatures::Int)::Node
    tree = Node(1.0f0)
    for i=1:length
        tree = appendRandomOp(tree, options, nfeatures)
    end
    return tree
end

