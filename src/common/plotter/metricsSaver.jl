

# Function loading dataframes from directory
function load_dataframes(directory)
    local dataframes = []
    for (root, dirs, files) in walkdir(directory)
        for file = joinpath.(root, files)
            local dataframe = CSV.read(file, DataFrame)
            push!(dataframes, dataframe)
        end
    end
    return dataframes
end

function load_metrics_structs(dataframes)
    local epochs = []
    local metrics_structs = []
    for dataframe in dataframes
        local predicted_classes = dataframe."predicted_class"
        local expected_classes = dataframe."expected_class"
        # println("predicted_classes: $(size(predicted_classes)), expected_classes: $(size(expected_classes)), number_of_classes: $(number_of_classes)")
        local metrics_struct = MetricsModule1.metrics(predicted_classes, expected_classes, number_of_classes)
        push!(metrics_structs, metrics_struct)

        local epoch = dataframe."epoch"[1]
        push!(epochs, epoch)
    end
    return [metrics_structs, epochs]
end

function get_mean_metrics(metrics_datastructs, getMetrics::Function)
    local mean_metrics = []
    for metrics_datastruct = metrics_datastructs
        local mean_metric = mean(getMetrics(metrics_datastruct) ./ length(getMetrics(metrics_datastruct)))
        push!(mean_metrics, mean_metric)
    end
    return mean_metrics
end

function get_metrics(metrics_datastructs, getMetrics::Function)
    local metrics = []
    for metrics_datastruct = metrics_datastructs
        push!(metrics, getMetrics(metrics_datastruct))
    end
    return metrics
end

function calculate_and_save_metrics()
    local train_dataframes = load_dataframes(train_dumps_path)
    local test_dataframes = load_dataframes(test_dumps_path)

    println("test_dataframes: $(size(test_dataframes))")

    # Function calculating metrics structs for dataframes
    local test_metrics_datastructs, test_epochs = load_metrics_structs(test_dataframes)
    local train_metrics_datastructs, train_epochs = load_metrics_structs(train_dataframes)

    metrics = Dict(
        "Accuracy" => Dict(
            "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.accuracy),
            "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.accuracy),
            "plot" => true
        ),
        "Precision" => Dict(
            "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.precision),
            "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.precision),
            "plot" => true
        ),
        "Recall" => Dict(
            "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.recall),
            "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.recall),
            "plot" => true
        ),
        "F1" => Dict(
            "train" => get_mean_metrics(train_metrics_datastructs, (md) -> md.f1),
            "test" => get_mean_metrics(test_metrics_datastructs, (md) -> md.f1),
            "plot" => true
        ),
        "Confussion matrix" => Dict(
            "plot" => true,
            "confussion_matrix" => true,
            "train" => get_metrics(train_metrics_datastructs, (md) -> md.confussion_matrix),
            "test" => get_metrics(test_metrics_datastructs, (md) -> md.confussion_matrix)
        ),
    )

    if (!isdir(results_path))
        mkdir(results_path)
    end

    json_string = JSON.json(metrics)
    open("$(results_path)/metrics.json", "w") do io
        println(io, json_string)
    end

    return [test_epochs, train_epochs, metrics]
end

# Deprecated function - use python script to plot confussion matrix
function save_confussion_matrix(confussion_matrix, filename, epochs::Int64, results_path::String, classes::Array{Any,1})
    println("confussion_matrix: $(confussion_matrix)")
    println("confussion_matrix: $(typeof(confussion_matrix))")
    println("===")
    for idx = 1:epochs
        local file_path = "$(results_path)/$(filename)_$(idx-1).png"
        local cm = confussion_matrix[idx]
        println("idx: $(idx), cm: $(cm)")
        PlotlyJS.savefig(
            PlotlyJS.plot(
                PlotlyJS.heatmap(
                    x=classes,
                    y=classes,
                    z=map(Int64, cm)',
                    series_annotations="series_ann",
                ),
                PlotlyJS.Layout(xaxis_side="top"),
            ),
            file_path
        )
        println("saving $(file_path) file")
        # savefig(file_path)
    end
end