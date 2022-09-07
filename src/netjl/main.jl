include("utils/loader.jl")
include("utils/argparser.jl")
include("../common/MetricsModule1.jl")

using Dates
using JSON

import NetworkModule
import WeightsModule

function makeTest(test_cases, weights, classes_quantity)
    test_results = NetworkModule.test(test_cases, weights)
    predicted_classes = getindex.(test_results, 2)
    expected_classes = getindex.(test_results, 3)
    metricsStruct = MetricsModule1.metrics(predicted_classes, expected_classes, classes_quantity)
    println("accuracy: $(metricsStruct.accuracy)")
    println("resut: $(NetworkModule.getStringOfSuccessPercentage(test_cases, weights))")
end

getCheckpointsPath = (current_train_directory::String) -> "$(current_train_directory)/checkpoints"
getCheckpointFilePath = (dataset, epoch::Int64, checkpointsPath::String) -> "$(checkpointsPath)/checkpoint_$(epoch).weights"
getTrainDumpsPath = (current_train_directory::String) -> "$(current_train_directory)/train_dumps"
getTrainDumpFilePath = (epoch::Int64, trainDumpsPath::String) -> "$(trainDumpsPath)/train_dump_$(epoch).csv"
getTestDumpsPath = (current_train_directory::String) -> "$(current_train_directory)/test_dumps"
getTestDumpFilePath = (epoch::Int64, testDumpsPath::String) -> "$(testDumpsPath)/test_dump_$(epoch).csv"


return

parsed_args = parse_commandline()

#
# configuration
#
dataset = getDataset(parsed_args)
lr = getLearningRate(parsed_args)
epochs = getEpochs(parsed_args)
###

datasetBase = dataset.createDatasetBase()
train = datasetBase.train
test = datasetBase.test
classes = datasetBase.classes

println("Use $(dataset.getName()) dataset")
println("train dataset size: $(size(train))")
println("test dataset size: $(size(test))")

#
# extract information about input and output from dataset
#
firstLayerWidth = size(train[1][1])[1]
outLayerWidth = size(classes)[1]

#
# create initial weights
#
initialWeights = NetworkModule.Weights(firstLayerWidth, outLayerWidth)

#
# Get basics file sture
#
default_output_folder = "src/netjl/outputs"
if (!isdir(default_output_folder))
    mkdir(default_output_folder)
end
now = Dates.now()
current_train_directory_name = getName(parsed_args)
current_train_directory = "$(default_output_folder)/$(current_train_directory_name)"


#
# metrics
#
if (parsed_args["metrics"] !== nothing)
    # Deprecated
    include("utils/metrics.jl")
    exit()
end

#
# continue_from_checkpoint
#
if (parsed_args["continue_from_checkpoint"] !== nothing)
    include("utils/continueFromCheckpoint.jl")
    exit()
end

#
# Prepare train files structure
#
begin
    if (!isdir(current_train_directory))
        mkdir(current_train_directory)
    end

    current_checkpoints_path = getCheckpointsPath(current_train_directory)
    if (!isdir(current_checkpoints_path))
        mkdir(current_checkpoints_path)
    end

    current_train_dumps_path = getTrainDumpsPath(current_train_directory)
    if (!isdir(current_train_dumps_path))
        mkdir(current_train_dumps_path)
    end

    current_test_dumps_path = getTestDumpsPath(current_train_directory)
    if (!isdir(current_test_dumps_path))
        mkdir(current_test_dumps_path)
    end
end

println("Start new training")
# save config file
config = Dict(
    "dataset" => dataset.getName(),
    "lr" => lr,
    "epochs" => epochs,
    "datetime" => Dates.format(now, "yyyy-mm-dd HH:MM:SS"),
    "directory" => current_train_directory_name,
    "classes" => classes,
)
json_string = JSON.json(config)
open("$(current_train_directory)/config.json", "a") do io
    println(io, json_string)
end

#
# save initial weight
#
WeightsModule.saveToFile(initialWeights, getCheckpointFilePath(dataset, 0, getCheckpointsPath(current_train_directory)))

#
# load initial weight
#
checkpoint = WeightsModule.loadFromFile(getCheckpointFilePath(dataset, 0, getCheckpointsPath(current_train_directory)))
weights = checkpoint.weights

#
# make test before training
#
println("\nTest with initial weights:")
makeTest(test, weights, length(classes))

#
# train
#
getTrainDump = (epoch::Int64) -> getTrainDumpFilePath(epoch, getTrainDumpsPath(current_train_directory))
getTestDump = (epoch::Int64) -> getTestDumpFilePath(epoch, getTestDumpsPath(current_train_directory))

NetworkModule.saveTestDump(weights, test, 0, getTestDump)

for epoch = 1:epochs
    NetworkModule.train(weights, train, test, epoch, lr, getTrainDump)

    #
    # make test after train
    #
    println("\nTest after training $(epoch) epoch:")
    makeTest(test, weights, length(classes))
    NetworkModule.saveTestDump(weights, test, epoch, getTestDump)

    #
    # save checkpoint
    #
    WeightsModule.saveToFile(weights, getCheckpointFilePath(dataset, epoch, getCheckpointsPath(current_train_directory)))
end

#
# Save train dump for last epoch
#
NetworkModule.saveTrainDump(weights, train, epochs + 1, lr, getTrainDump)


