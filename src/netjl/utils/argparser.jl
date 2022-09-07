using ArgParse
using Dates

# datasets
include("../../common/datasets/DigitMNIST.jl")
include("../../common/datasets/FashionMNIST.jl")
include("../../common/datasets/Iris.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--continue_from_checkpoint", "-c"
        help = "Continue training from selected checkpoint"

        "--metrics", "-m"
        help = "Run test on checkpoint"

        "--dataset", "-d"
        help = "Select dataset"
        default = "DigitMNIST"

        "--learning_rate", "-l"
        help = "Learning rate"
        arg_type = Float64
        default = 0.4

        "--epochs", "-e"
        help = "Quantity of epochs"
        arg_type = Int
        default = 5

        "--plot", "-p"
        help = "Plot metrics for dumps"

        "--name", "-n"
        help = "Name of the trainig"
        default = Dates.format(Dates.now(), "yyyymmddHHMMSS")
    end

    return parse_args(s)
end



function getDataset(parsed_args)
    if (parsed_args["dataset"] === "Iris")
        return Iris
    elseif (parsed_args["dataset"] === "FashionMNIST")
        return FashionMNIST
    end
    return DigitMNIST
end

function getLearningRate(parsed_args)
    return parsed_args["learning_rate"]
end

function getEpochs(parsed_args)
    return parsed_args["epochs"]
end

function getName(parsed_args)
    return parsed_args["name"]
end