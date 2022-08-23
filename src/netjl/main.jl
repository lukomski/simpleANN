
modules_path = "$(pwd())/src/netjl/modules"
common_path = "$(pwd())/src/common"
datasets_path = "$(pwd())/src/common/datasets"
push!(LOAD_PATH, modules_path)
push!(LOAD_PATH, datasets_path)
push!(LOAD_PATH, common_path)

println("modules_path: $(modules_path)")
println("datasets_path: $(datasets_path)")

# datasets
import DigitMNIST
import FashionMNIST
import Iris

# network
import NetworkModule
import WeightsModule
using NetworkModule
using Metrics

function makeTest(test_cases, weights, classes_quantity)
    test_results = NetworkModule.test(test_cases, weights)
    predicted_classes = getindex.(test_results, 2)
    expected_classes = getindex.(test_results, 3)
    metricsStruct = metrics(predicted_classes, expected_classes, classes_quantity)
    println("accuracy: $(metricsStruct.accuracy)")
    println("resut: $(NetworkModule.getStringOfSuccessPercentage(test_cases, weights))")
end

# configure dataset
dataset = Iris

datasetBase = dataset.createDatasetBase()
train = datasetBase.train
test = datasetBase.test
classes = datasetBase.classes

# display selected configuration
println("Use $(dataset.getName()) dataset")

println("train dataset size: $(size(train))")
println("test dataset size: $(size(test))")

# extract information about input and output from dataset
firstLayerWidth = size(train[1][1])[1]
outLayerWidth = size(classes)[1]

# create initial weights
initialWeights = NetworkModule.Weights(firstLayerWidth, outLayerWidth)

# go to folder with checkpoints
cd("src/netjl")
defaultWeightsFolder = "checkpoints"
if (!isdir(defaultWeightsFolder))
    mkdir(defaultWeightsFolder)
end
cd(defaultWeightsFolder)

# save initial weight
WeightsModule.saveToFile(initialWeights, "weight.0")

# load initial weight
checkpoint = WeightsModule.loadFromFile("weight.0")
weights = checkpoint.weights

# make test before training
println("\nTest with initial weights:")
makeTest(test, weights, length(classes))

epochs = 1
# train one epoch
NetworkModule.train(weights, train, test, epochs)

# make test after train
println("\nTest after training one epoch:")
makeTest(test, weights, length(classes))

# save checkpoint after first epoch
WeightsModule.saveToFile(weights, "weight.$(epochs)")
