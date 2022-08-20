module Network

    #
    # Structures
    #

    abstract type GraphNode end
    abstract type Operator <: GraphNode end

    struct Constant{T} <: GraphNode
        output :: T
    end

    mutable struct Variable <: GraphNode
        output :: Any
        gradient :: Any
        name :: String
        Variable(output; name="?") = new(output, nothing, name)
    end

    mutable struct ScalarOperator{F} <: Operator
        inputs :: Any
        output :: Any
        gradient :: Any
        name :: String
        ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
    end

    mutable struct BroadcastedOperator{F} <: Operator
        inputs :: Any
        output :: Any
        gradient :: Any
        name :: String
        BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
    end

    function printGraph(graph)
        println("\ngraph ( $(length(graph)) ):")
        for node_idx in 1:length(graph)
            print("$(node_idx):")
            show(graph[node_idx])
    #         print(", output: ", graph[node_idx].output != nothing && length(graph[node_idx].output) > 10 ? "huge" : graph[node_idx].output)
                
    #         println("typeof(graph[node_idx]):", typeof(graph[node_idx]))
    #         if typeof(graph[node_idx]) != Constant && (graph[node_idx].gradient == nothing || length(graph[node_idx].gradient) < 10)
    #             print(", gradient: $(graph[node_idx].gradient)")
    #         end
            
            println("")
        end
        println("END OF graph\n")
    end


    #
    # Graph
    #

    function visit(node::GraphNode, visited, order)
        if node ∈ visited
        else
            push!(visited, node)
            push!(order, node)
        end
        return nothing
    end
        
    function visit(node::Operator, visited, order)
        if node ∈ visited
        else
            push!(visited, node)
            for input in node.inputs
                visit(input, visited, order)
            end
            push!(order, node)
        end
        return nothing
    end
    
    function topological_sort(head::GraphNode)
        visited = Set()
        order = Vector()
        visit(head, visited, order)
        return order
    end

    #
    # Operations
    #

    import Base: show, summary
    show(io::IO, x::ScalarOperator{F}) where {F} = begin
        print(io, "op2 ", x.name, "(", F, ")")
        print(", inputs1 ($(length(x.inputs)))")
        println()
    end
    show(io::IO, x::BroadcastedOperator{F}) where {F} = begin
        print(io)
        print("op1.", x.name, "(", F, ")")
        print(", inputs2 ($(length(x.inputs)))")
        print(", output ($(typeof(x.output))): ", x.output == nothing || length(x.output) <= 10 ? x.output : "$(x.output[1:2])...")
        print(", gradient ($(typeof(x.gradient)): ", x.gradient == nothing || length(x.gradient) <= 10 ? x.gradient : "$(x.gradient[1:2])...")
        println()
    end
    show(io::IO, x::Constant) = print(io, "const ", x.output)
    show(io::IO, x::Variable) = begin
        print(io, "var ", x.name);
        print(io, "\n ┣━ ^ ");
            summary(io, x.output);
            print(", x.output: ", x.output == nothing || length(x.output) <= 10 ? x.output : "$(x.output[1:2])...")
        print(io, "\n ┗━ ∇ "); 
            summary(io, x.gradient)
            print(", x.gradient: ", x.gradient == nothing || length(x.gradient) <= 10 ? x.gradient : "$(x.gradient[1:2])...")
        println()
    end



    reset!(node::Constant) = nothing
    reset!(node::Variable) = node.gradient = nothing
    reset!(node::Operator) = node.gradient = nothing

    compute!(node::Constant) = nothing
    compute!(node::Variable) = nothing
    compute!(node::Operator) =
        node.output = forward(node, [input.output for input in node.inputs]...)

    function forward!(order::Vector)
        for node in order
            compute!(node)
            reset!(node)
        end
        return last(order).output
    end

    update!(node::Constant, gradient) = nothing
    update!(node::GraphNode, gradient) =
        if isnothing(node.gradient)
            node.gradient = gradient
        else
            node.gradient .+= gradient
    end

    function backward!(order::Vector; seed=1.0)
        result = last(order)
        result.gradient = seed
        @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
        for node in reverse(order)
            backward!(node)
        end
        return nothing
    end

    function backward!(node::Constant) end
    function backward!(node::Variable) end
    function backward!(node::Operator)
        inputs = node.inputs
        gradients = backward(node, [input.output for input in inputs]..., node.gradient)
        for (input, gradient) in zip(inputs, gradients)
            update!(input, gradient)
        end
        return nothing
    end

    import Base: ^
    ^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
    forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
    backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

    import Base: *
    import LinearAlgebra: mul!, diagm
    # x * y (aka matrix multiplication)
    *(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
    forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
    backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

    # x .* y (element-wise multiplication)
    Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
    forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
    backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * g, Jy' * g)
    end

    Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
    forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
    backward(::BroadcastedOperator{typeof(-)}, x, y, g) = begin
        tuple(g,-g)
    end
    Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
    forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
    backward(::BroadcastedOperator{typeof(+)}, x, y, g) = begin
        return tuple(g, g)
    end

    import Base: sum
    sum(x::GraphNode) = BroadcastedOperator(sum, x)
    forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
    backward(::BroadcastedOperator{typeof(sum)}, x, g) = let 
    #     println("\n[SUM]")
        𝟏 = ones(length(x))
        J = 𝟏'
        tuple(J' * g)
    end

    Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
    forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
    backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
        𝟏 = ones(length(node.output))
        @assert 0 ∉ y "Add some eposilon not to divide by 0 y: $(y)"
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^2)
        tuple(Jx' * g, Jy' * g)
    end
    backward(node::BroadcastedOperator{typeof(/)}, x, y, g) = let
        𝟏 = ones(length(node.output))
        @assert 0 ∉ y "Add some eposilon not to divide by 0 y: $(y)"
        Jx = diagm(𝟏 ./ y)
        Jy = -x .* diagm(𝟏 ./ (y.^ 2))
        tuple(Jx' * g, Jy' * g)
    end

    #
    # More operations
    #

    import Base:zero, -, exp

    Base.Broadcast.broadcasted(-, x::GraphNode) = BroadcastedOperator(-, x)
    -(x::BroadcastedOperator{typeof(-)}) = return BroadcastedOperator(-, x)
    forward(::BroadcastedOperator{typeof(-)}, x) = return .-x
    backward(::BroadcastedOperator{typeof(-)}, x, g) = return tuple(-g)


    Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
    exp(x::BroadcastedOperator{typeof(-)}) = return BroadcastedOperator(exp, x)
    forward(::BroadcastedOperator{typeof(exp)}, x) = begin
        return exp.(x)
    end
    backward(::BroadcastedOperator{typeof(exp)}, x, g) = return tuple(g .* exp.(x))



    # dont use this except forward, WHY? 
    Base.Broadcast.broadcasted(^, x::GraphNode) = BroadcastedOperator(^, x)
    ^(x::BroadcastedOperator{typeof(^)}) = return BroadcastedOperator(^, x)
    forward(::BroadcastedOperator{typeof(^)}, x, n) = return x .^ n
    backward(::BroadcastedOperator{typeof(^)}, x, n, g) = begin
    #     println("\n[^]")
        return tuple(g .* n .* x .^ (n.-1), g .* log.(abs.(x)) .* x .^ n)
    end

    #
    # Functions
    #

    sigmoid(x) = return Constant(1.0) ./ (Constant(1.0) .+ exp(.-x))
    softmax(x) = exp.(x) ./ sum(exp.(x))


    #
    # Definition of network
    #

    using MLDatasets
    # Init
    struct Weights
        Wh1::Variable
        Wh2::Variable
        Wo::Variable
        Weights() = new(
            Variable(randn(32,784), name="Wh1"),
            Variable(randn(32,32), name="Wh2"),
            Variable(randn(10,32), name="Wo"),
        )
    end

    onehot_to_digit(y::Vector) = argmax(y) - 1

    function net(x, weights::Weights)
        x1 = sigmoid(weights.Wh1 * x);
        x2 = sigmoid(weights.Wh2 * x1);
        ŷ = sigmoid(weights.Wo * x2);
        return ŷ
    end

    function predict_digit(x, weights::Weights)
        ŷ = net(x, weights)
        o = topological_sort(ŷ)    
        forward!(o)
        onehot_to_digit(ŷ.output)
    end

    function success_percentage(data_set, weights)
        return string("Percentage of correctly classified images is: ", 
            sum([ predict_digit(Constant(x[1]), weights) == 
            argmax(x[2]) - 1 ? 1 : 0 for x in data_set])/length(data_set)*100, " %")
    end

    function getTestDataset()
        test_x, test_y = MNIST.testdata(Float64);
        
        X = []; 
        Y = []; 

        for i = 1 : 10000
            push!(X, reshape(test_x[:,:,i],784));
            y = zeros(10);
            y[test_y[i] + 1] = 1.0; 
            push!(Y,y);
        end

        test_data = [x for x in zip(X,Y)]; 
    end

    function update_weights!(x, weights::Weights, y, lr=0.4)
        ŷ = net(x, weights)
        E = sum(Constant(0.5).*((y .- ŷ).^Constant(2)));
        o = topological_sort(E)
        forward!(o)
        backward!(o)
        weights.Wh1.output -= lr .* weights.Wh1.gradient
        weights.Wh2.output -= lr .* weights.Wh2.gradient
        weights.Wo.output -= lr .* weights.Wo.gradient
        return nothing
    end 

    # #
    # # Initialization of weights
    # #

    # weights = Weights()
    # test_dataset = getTestDataset()
    # success_percentage(test_dataset, weights)

    # #
    # # Train
    # #

    # # Trenowanie zbioru testowego
    # for i=1:1000
    #     x = Constant(test_dataset[i][1])
    #     y = Constant(test_dataset[i][2])
    #     update_weights!(x,weights, y)
    # end
    # success_percentage(test_dataset, weights)
end