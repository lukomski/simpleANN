###
# Network module
###

using Statistics

get_class(y::Vector) = argmax(y) - 1

function predict(x, Wh1, bh1, Wh2, bh2, Wo, bo)
    x1 = (Wh1 * x .+ bh1);
    x̂1 = sigmoid(x1);
    x2 = (Wh2 * x̂1 .+ bh2);
    x̂2 =  sigmoid(x2);
    x3 = (Wo * x̂2 .+ bo);
    ŷ = sigmoid(x3);
    graph = topological_sort(ŷ)
    forward!(graph)
    get_class(ŷ.output)
end

function accuracy(data_set)
    return sum([ predict(Constant(x[1]), Wh1, bh1, Wh2, bh2, Wo, bo) == get_class(x[2]) ? 1 : 0 for x in data_set])/length(data_set)*100
end

function metrics(data_set, number_of_classes)

    prediction_list_all = zeros(number_of_classes)
    prediction_list_correct = zeros(number_of_classes)
    correct_list_all = zeros(number_of_classes)
    #println(prediction_list_all)
    #println(prediction_list_correct)
    #println(correct_list_all)

    for i in data_set
        prediction = predict(Constant(i[1]), Wh1, bh1, Wh2, bh2, Wo, bo)
        correct_class = get_class(i[2])

        prediction_list_all[prediction + 1] += 1
        correct_list_all[correct_class + 1] += 1
        if (prediction == correct_class)
            prediction_list_correct[prediction + 1] += 1
        end
    end

    println(prediction_list_all)
    println(prediction_list_correct)
    println(correct_list_all)

    data_set_size = length(data_set)
    precision = mean(prediction_list_correct./prediction_list_all)
    recall = mean(prediction_list_correct./correct_list_all)
    println("Accuracy: ", sum(prediction_list_correct./data_set_size)*100, "%")
    println("Precision: ", precision * 100, "%")
    println("Recall: ", recall * 100, "%")
    println("F1: ", 2*(precision*recall/(precision+recall)))

    return nothing
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