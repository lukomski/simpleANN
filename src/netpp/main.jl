####
#   Paweł Podgórski
#   
#   
####

push!(LOAD_PATH, "$(pwd())/../common/datasets")
import DigitMNIST
import FashionMNIST
import Iris

############### CONFIGURATION
dataset = FashionMNIST
epochs = 5
lr = 0.4
###############

println("Program started!")

import LinearAlgebra

abstract type GraphNode end
abstract type Operator <: GraphNode end
struct Constant{T} <: GraphNode
    output :: T
end
mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output;name="?") = new(output, name) 
end
mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any 
    gradient :: Any 
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name) #konstruktor
end
mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any 
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name) #konstruktor
end

function visit(node::GraphNode, visited, order)
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
function visit(node::Operator, visited, order)
    if node ∉ visited
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

reset!(node::Constant) = nothing 
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing
compute!(node::Constant) = nothing 
compute!(node::Variable) = nothing 
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient 
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

import Base: sin, cos 
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

cos(x::GraphNode) = ScalarOperator(cos, x)
forward(::ScalarOperator{typeof(cos)}, x) = return cos(x)
backward(::ScalarOperator{typeof(cos)}, x, g) = tuple(-g * sin(x))

import Base: exp
exp(x::GraphNode) = ScalarOperator(exp, x)
forward(::ScalarOperator{typeof(exp)},x) = return exp(x)
backward(::ScalarOperator{typeof(exp)},x,g) = tuple(g*exp(x))

import Base: -, +, *, / 
-(x::GraphNode) = ScalarOperator(-,x)
forward(::ScalarOperator{typeof(-)},x) = return -x
backward(::ScalarOperator{typeof(-)},x,g) = tuple(-g)

-(x::GraphNode, y::GraphNode) = ScalarOperator(-,x,y)
forward(::ScalarOperator{typeof(-)},x,y) = x - y
backward(::ScalarOperator{typeof(-)},x,y,g) = tuple(g,-g)

+(x::GraphNode, y::GraphNode) = ScalarOperator(+,x,y)
forward(::ScalarOperator{typeof(+)},x,y) = x + y
backward(::ScalarOperator{typeof(+)},x,y,g) = tuple(g,g)

*(x::GraphNode, y::GraphNode) = ScalarOperator(*,x,y)
forward(::ScalarOperator{typeof(*)},x,y) = x * y
backward(::ScalarOperator{typeof(*)},x,y,g) = tuple(g*y, g*x)


/(x::GraphNode, y::GraphNode) = ScalarOperator(/,x,y)
forward(::ScalarOperator{typeof(/)},x,y) = x / y
backward(::ScalarOperator{typeof(/)},x,y,g) = tuple(g/y,g*-x/y/y)

import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication) - mnozenie macierzy przez wektor
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# x .* y (element-wise multiplication) - mnozenie element po  elemencie
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = LinearAlgebra.diagm(y .* 𝟏)
    Jy = LinearAlgebra.diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(^, x::GraphNode, n::GraphNode) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = return x .^ n
backward(node::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(g .* n .* x .^ (n-1), g .* log.(abs.(x)) .* x .^ n)


Base.Broadcast.broadcasted(exp, x::GraphNode) = BroadcastedOperator(exp, x)
forward(::BroadcastedOperator{typeof(exp)}, x) = return exp.(x)
backward(node::BroadcastedOperator{typeof(exp)}, x, g) = tuple(g.*exp.(x))

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g)

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = LinearAlgebra.diagm(isless.(y, x))
    Jy = LinearAlgebra.diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

# using MLDatasets

# train_x, train_y = MNIST.traindata(Float64);
# test_x, test_y = MNIST.testdata(Float64);

# X = []; 
# Y = []; 

# for i = 1 : 60000
#     @views push!(X, reshape(train_x[:,:,i],784));
#     y = zeros(10);
#     y[train_y[i] + 1] = 1.0; 
#     push!(Y,y);
# end

# train_data = [x for x in zip(X,Y)];

# X = []; 
# Y = []; 

# for i = 1 : 10000
#     @views push!(X, reshape(test_x[:,:,i],784));
#     y = zeros(10);
#     y[test_y[i] + 1] = 1.0; 
#     push!(Y,y);
# end

# test_data = [x for x in zip(X,Y)];

datasetBase = dataset.createDatasetBase()
train_data = datasetBase.train
test_data = datasetBase.test
classes = datasetBase.classes

println("Use $(dataset.getName()) dataset")
println("train dataset size: $(size(train_data))")
println("test dataset size: $(size(test_data))")

first_layer_width = size(train_data[1][1])[1]
out_layer_width = size(classes)[1]

Wh1 = Variable(randn(32, first_layer_width), name="Wh1")
bh1 = Variable(randn(32), name="bh1")
Wh2 = Variable(randn(32,32), name="Wh2")
bh2 = Variable(randn(32), name="bh2")
Wo = Variable(randn(out_layer_width,32), name="Wo")
bo = Variable(randn(out_layer_width), name="bo")

get_digit(y::Vector) = argmax(y) - 1

function predict_digit(x, Wh1, bh1, Wh2, bh2, Wo, bo)
    x1 = (Wh1 * x .+ bh1);
    x̂1 = sigmoid(x1);
    x2 = (Wh2 * x̂1 .+ bh2);
    x̂2 =  sigmoid(x2);
    x3 = (Wo * x̂2 .+ bo);
    ŷ = sigmoid(x3);
    graph = topological_sort(ŷ)
    forward!(graph)
    get_digit(ŷ.output)
end

function success_percentage(data_set)
    return string("Percentage of correctly classified images is: ", 
        sum([ predict_digit(Constant(x[1]), Wh1, bh1, Wh2, bh2, Wo, bo) == get_digit(x[2]) ? 1 : 0 for x in data_set])/length(data_set)*100, " %")
end

one = Constant(1.0)
minus_one = Constant(-1.0)
two = Constant(2.0)
half = Constant(0.5)

linear(x) = x
sigmoid(z) = return (one .+ exp.(z)) .^ minus_one
mean_squared_loss(y, ŷ) = return half .* ((y .- ŷ) .^ two)

function train!(x, Wh1, bh1, Wh2, bh2, Wo, bo, y, lr, loss)
    x1 = (Wh1 * x .+ bh1);
    x̂1 = sigmoid(x1);
    x2 = (Wh2 * x̂1 .+ bh2);
    x̂2 =  sigmoid(x2);
    x3 = (Wo * x̂2 .+ bo);
    ŷ = sigmoid(x3);
    E = sum(mean_squared_loss(y, ŷ));
    graph = topological_sort(E)
    forward!(graph)
    loss_node = last(graph)
    loss += loss_node.output
    #print("loss", loss, "\n")
    backward!(graph)
    Wh1.output -= lr .* Wh1.gradient
    bh1.output -= lr .* bh1.gradient
    Wh2.output -= lr .* Wh2.gradient
    bh2.output -= lr .* bh2.gradient
    Wo.output -= lr .* Wo.gradient
    bo.output -= lr .* bo.gradient
    return nothing
end 

loss_list = []

for epoch in 1:epochs
    loss = 0.0
    for i = 1:length(train_data)
        x = Constant(train_data[i][1])
        y = Constant(train_data[i][2])
        train!(x, Wh1, bh1, Wh2, bh2, Wo, bo, y, lr, loss)
        push!(loss_list, loss)
    end
    println("Epoch ", epoch, " Loss is: ", loss, "\n")
end

println("Result:")
println("Wynik danych testowych: ", success_percentage(test_data))
println("Wynik danych treningowych: ", success_percentage(train_data))