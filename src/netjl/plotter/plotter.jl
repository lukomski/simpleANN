include("argparser.jl")
include("../../common/Metrics.jl")

parsed_args = parse_commandline()

using Plots
using CSV
using DataFrames
using Statistics
using JSON

default_outputs_path = "./src/netjl/outputs"
output_path = "$(default_outputs_path)/$(parsed_args["directory"])"
train_dumps_path = "$(output_path)/train_dumps"
test_dumps_path = "$(output_path)/test_dumps"
results_path = "$(output_path)/results"


# Function loading dataframes from directory
load_dataframes = (directory) -> begin
    local dataframes = []
    for (root, dirs, files) in walkdir(directory)
        for file = joinpath.(root, files)
            local dataframe = CSV.read(file, DataFrame)
            push!(dataframes, dataframe)
        end
    end
    return dataframes
end

train_dataframes = load_dataframes(train_dumps_path)
test_dataframes = load_dataframes(test_dumps_path)

# Function calculating metrics structs for dataframes

load_metrics_structs = (dataframes) -> begin
    local epochs = []
    local metrics_structs = []
    for dataframe = train_dataframes
        local predicted_classes = dataframe."predicted_class"
        local expected_classes = dataframe."expected_class"
        local metrics_struct = Metrics.metrics(predicted_classes, expected_classes, 10)
        push!(metrics_structs, metrics_struct)

        local epoch = dataframe."epoch"[1]
        push!(epochs, epoch)
    end
    return [metrics_structs, epochs]
end

train_metrics_datastructs, train_epochs = load_metrics_structs(train_dataframes)
test_metrics_datastructs, test_epochs = load_metrics_structs(test_dataframes)

get_mean_metrics = (metrics_datastructs, getMetrics::Function) -> begin
    local mean_metrics = []
    for metrics_datastruct = metrics_datastructs
        local mean_metric = mean(getMetrics(metrics_datastruct) ./ length(getMetrics(metrics_datastruct)))
        push!(mean_metrics, mean_metric)
    end
    return mean_metrics
end

metrics = Dict(
    "Accuracy" => Dict(
        "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.accuracy),
        "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.accuracy)
    ),
    "Precision" => Dict(
        "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.precision),
        "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.precision)
    ),
    "Recall" => Dict(
        "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.recall),
        "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.recall)
    ),
    "F1" => Dict(
        "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.f1),
        "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.f1)
    ),
)

if (!isdir(results_path))
    mkdir(results_path)
end

json_string = JSON.json(metrics)
open("$(results_path)/metrics.json", "a") do io
    println(io, json_string)
end

for (key, value) in metrics
    x = train_epochs
    y = value["train"]
    z = value["test"]

    #use GR module alternatively may use plotlyjs()
    gr()

    plot(x, [y, z], title=key, label=["Train" "Test"], lw=[6 3], legend=:bottomright, seriestype=[:line :path], seriesalpha=[1 1])

    # gui()
    xlabel!("Epochs")
    ylabel!("Accuracy")

    println("saving $(results_path)/$(key).png file")
    savefig("$(results_path)/$(key).png")
end
