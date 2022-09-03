module NetworkModule
using GraphModule
using WeightsModule
export net, sigmoid, softmax

#
# Functions
#

sigmoid(x) = return Constant(1.0) ./ (Constant(1.0) .+ exp(.-x))
softmax(x) = exp.(x) ./ sum(exp.(x))


#
# Definition of network
#

# Init


onehot_to_digit(y::Vector) = argmax(y) - 1

function net(x, weights::Weights)
    x1 = sigmoid(weights.Wh1 * x)
    x2 = sigmoid(weights.Wh2 * x1)
    ŷ = sigmoid(weights.Wo * x2)
    return ŷ
end

function predict_digit(x, weights::Weights)
    ŷ = net(x, weights)
    o = GraphModule.topological_sort(ŷ)
    GraphModule.forward!(o)
    onehot_to_digit(ŷ.output)
end

function success_percentage(data_set, weights)
    return sum([predict_digit(Constant(x[1]), weights) ==
                argmax(x[2]) - 1 ? 1 : 0 for x in data_set]) / length(data_set) * 100
end

function getStringOfSuccessPercentage(data_set, weights)
    return string("Percentage of correctly classified images is: ",
        success_percentage(data_set, weights), " %")
end

using JLD2
using JSON

function update_weights!(x, weights::Weights, y, lr=0.4, getDump::Function=() -> nothing, epoch=nothing, withUpdateWeights::Bool=true)
    ŷ = net(x, weights)

    E = sum(Constant(0.5) .* ((y .- ŷ) .^ Constant(2)))
    o = GraphModule.topological_sort(E)
    GraphModule.forward!(o)
    GraphModule.backward!(o)

    # dump
    dump = getDump(epoch - 1) # dump is calculated for previous epoch. After calculating result it updates weights.
    if (dump !== nothing)
        if (!isfile(dump))
            # save headers
            headers = ["epoch", "expected_class", "predicted_class"]
            open(dump, "a") do io
                println(io, join(headers, ';'))
            end
        end
        open(dump, "a") do io
            println(io, join([epoch; onehot_to_digit(y.output); onehot_to_digit(ŷ.output)], ';'))
        end
    end

    if (withUpdateWeights)
        weights.Wh1.output -= lr .* weights.Wh1.gradient
        weights.Wh2.output -= lr .* weights.Wh2.gradient
        weights.Wo.output -= lr .* weights.Wo.gradient
    end
    return nothing
end

function train(weights::Weights, train, test, epoch::Int, lr::Float64, getDump=nothing)
    # for epoch = 1:epochs
    for i = 1:size(train)[1]
        x = Constant(train[i][1])
        y = Constant(train[i][2])
        update_weights!(x, weights, y, lr, getDump, epoch)
    end
    # end
    return weights
end

function saveTrainDump(weights::Weights, train, epoch::Int, lr::Float64, getDump=nothing)
    for i = 1:size(train)[1]
        x = Constant(train[i][1])
        y = Constant(train[i][2])
        withUpdateWeights = false
        update_weights!(x, weights, y, lr, getDump, epoch, withUpdateWeights)
    end
end

function saveTestDump(weights::Weights, test, epoch::Int, getDump::Function)
    test_results = NetworkModule.test(test, weights)
    predicted_classes = getindex.(test_results, 2)
    expected_classes = getindex.(test_results, 3)
    dump = getDump(epoch)

    headers = ["epoch", "expected_class", "predicted_class"]
    open(dump, "a") do io
        println(io, join(headers, ';'))
        for idx = 1:size(predicted_classes)[1]
            expected_class = expected_classes[idx]
            predicted_class = predicted_classes[idx]
            println(io, join([epoch; expected_class; predicted_class], ';'))
        end
    end
end

function test(data_sets, weights::Weights)
    predicted_classes = []
    expected_classes = []
    for x in data_sets
        predicted_class = predict_digit(Constant(x[1]), weights)
        expected_class = argmax(x[2]) - 1
        push!(predicted_classes, predicted_class)
        push!(expected_classes, expected_class)
    end
    return zip(data_sets, predicted_classes, expected_classes)
end
end