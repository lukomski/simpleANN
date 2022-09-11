include("./argparser.jl")
include("../MetricsModule1.jl")
include("./metricsSaver.jl")

parsed_args = parse_commandline()

using Plots
import PlotlyJS
using CSV
using DataFrames
using Statistics
using JSON

default_outputs_path = "./src/$(parsed_args["net"])/outputs"
output_path = "$(default_outputs_path)/$(parsed_args["directory"])"
train_dumps_path = "$(output_path)/train_dumps"
test_dumps_path = "$(output_path)/test_dumps"
results_path = "$(output_path)/results"
config_file_path = "$(output_path)/config.json"

config = JSON.parsefile(config_file_path)
classes = config["classes"]
tests = config["tests"]
number_of_classes = length(classes)

metrics_sets = []
for test in tests
    push!(metrics_sets, calculate_metrics(test))
end

metrics = calculate_mean_metrics(metrics_sets)
train_epochs = metrics_sets[1].train_epochs

vvv = []

for (key, value) in metrics
    if (!haskey(value, "plot"))
        continue
    end
    if (haskey(value, "confussion_matrix"))
        # omit confussion matrixes - python script does the job
        continue
    end

    x = train_epochs
    y = value["train"]
    z = value["test"]

    gr() # gr() or plotlyjs()

    plot(x, [y, z], title=key, label=["Train" "Test"], lw=[6 3], legend=:bottomright, seriestype=[:line :path], seriesalpha=[1 1])

    # gui()
    xlabel!("Epochs")
    ylabel!(key)

    println("saving $(results_path)/$(key).png file")
    savefig("$(results_path)/$(key).png")
end
