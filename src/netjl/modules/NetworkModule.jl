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

function update_weights!(x, weights::Weights, y, lr=0.4)
    ŷ = net(x, weights)
    E = sum(Constant(0.5) .* ((y .- ŷ) .^ Constant(2)))
    o = GraphModule.topological_sort(E)
    GraphModule.forward!(o)
    GraphModule.backward!(o)
    weights.Wh1.output -= lr .* weights.Wh1.gradient
    weights.Wh2.output -= lr .* weights.Wh2.gradient
    weights.Wo.output -= lr .* weights.Wo.gradient
    return nothing
end

function train(weights::Weights, train, test, epochs::Int)
    for epoch = 1:epochs
        for i = 1:size(train)[1]
            x = Constant(train[i][1])
            y = Constant(train[i][2])
            update_weights!(x, weights, y)
        end
    end
    return weights
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