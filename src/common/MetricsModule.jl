module MetricsModule
export metrics, MetricsStruct
using Statistics

mutable struct MetricsStruct
    accuracy::Array{Float64,1}
    precision::Array{Float64,1}
    recall::Array{Float64,1}
    f1::Array{Float64,1}
    MetricsStruct(accuracy, precision, recall, f1) = new(accuracy, precision, recall, f1)
end

function metrics(predicted_classes::Array{Int64,1}, expected_classes::Array{Int64,1}, number_of_classes::Int64)
    data_set_size = length(predicted_classes)
    prediction_list_all = zeros(number_of_classes)
    prediction_list_correct = zeros(number_of_classes)
    correct_list_all = zeros(number_of_classes)
    for i = 1:data_set_size
        prediction = predicted_classes[i]
        correct_class = expected_classes[i]

        prediction_list_all[prediction+1] += 1
        correct_list_all[correct_class+1] += 1
        if (prediction == correct_class)
            prediction_list_correct[prediction+1] += 1
        end
    end

    precision = prediction_list_correct ./ prediction_list_all
    recall = prediction_list_correct ./ correct_list_all

    accuracy = prediction_list_correct ./ data_set_size

    f1 = 2 .* (precision .* recall ./ (precision .+ recall))

    metrics_struct = MetricsStruct(
        accuracy,
        precision,
        recall,
        f1
    )

    return metrics_struct
end
end