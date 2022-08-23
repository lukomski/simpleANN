###
# Network module
###

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

function precision(data_set)
    precision_list = []


    for i in train_data
        prediction = predict(Constant(i[1]), Wh1, bh1, Wh2, bh2, Wo, bo)
        correct_class = get_class(i[2])


    end
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