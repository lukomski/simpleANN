module MetricsModule1
export metrics, MetricsStruct
using Statistics
import Metrics

mutable struct MetricsStruct
    accuracy::Float64
    precision::Float64
    recall::Float64
    f1::Float64
    confussion_matrix::Any
    MetricsStruct(accuracy, precision, recall, f1, confussion_matrix) = new(accuracy, precision, recall, f1, confussion_matrix)
end

function metrics(predicted_classes::Array{Int64,1}, expected_classes::Array{Int64,1}, number_of_classes::Int64)
    local y_true = Metrics.onehot_encode(expected_classes, 0:number_of_classes)
    local y_pred = Metrics.onehot_encode(predicted_classes, 0:number_of_classes)

    local precision = Metrics.precision(y_pred, y_true)

    local accuracy = Metrics.categorical_accuracy(y_pred, y_true)

    local recall = Metrics.recall(y_pred, y_true)
    local f1 = 2 .* (precision .* recall ./ (precision .+ recall))
    local confussion_matrix = Metrics.confusion_matrix(y_pred, y_true)

    return MetricsStruct(
        accuracy,
        precision,
        recall,
        f1,
        confussion_matrix
    )
end
end