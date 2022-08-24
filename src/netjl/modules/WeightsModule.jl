module WeightsModule
using GraphModule
export Weights
export loadFromFile


struct Weights
    Wh1::Variable
    Wh2::Variable
    Wo::Variable
    Weights(input_length::Int64, class_number::Int64) = new(
        Variable(randn(32, input_length), name="Wh1"),
        Variable(randn(32, 32), name="Wh2"),
        Variable(randn(class_number, 32), name="Wo"),
    )
end
import Dates

struct Checkpoint
    weights::Weights
    datetime::Any
    Checkpoint(weights) = new(
        weights,
        Dates.Time(Dates.now())
    )
    Checkpoint(weights, datetime) = new(
        weights,
        datetime
    )
end

using JLD2
function saveToFile(weights::Weights, destination::String)
    checkpoint = Checkpoint(weights)
    @save destination checkpoint
end

function loadFromFile(source::String)
    @load source checkpoint
    checkpoint = Checkpoint(checkpoint.weights, checkpoint.datetime)
    return checkpoint
end
end