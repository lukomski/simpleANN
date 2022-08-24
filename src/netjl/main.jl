include("utils/loader.jl")
include("utils/argparser.jl")
using Dates

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

parsed_args = parse_commandline()

#
# configuration
#
dataset = FashionMNIST
lr = 0.4
epochs = 5
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
# Prepare file structure
#
default_checkpoint_folder = "src/netjl/checkpoints"
if (!isdir(default_checkpoint_folder))
    mkdir(default_checkpoint_folder)
end
current_train_folder = Dates.format(Dates.now(), "yyyymmddHHMMSS")
current_checkpoint_folder = "$(default_checkpoint_folder)/$(current_train_folder)"

#
# metrics
#
if (parsed_args["metrics"] !== nothing)
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

println("Start new training")
mkdir(current_checkpoint_folder)


#
# save initial weight
#
WeightsModule.saveToFile(initialWeights, "$(current_checkpoint_folder)/weight.0")

#
# load initial weight
#
checkpoint = WeightsModule.loadFromFile("$(current_checkpoint_folder)/weight.0")
weights = checkpoint.weights

#
# make test before training
#
println("\nTest with initial weights:")
makeTest(test, weights, length(classes))

#
# train
#
for epoch = 1:epochs
    NetworkModule.train(weights, train, test, epochs, lr)

    #
    # make test after train
    #
    println("\nTest after training $(epoch) epoch:")
    makeTest(test, weights, length(classes))

    #
    # save checkpoint
    #
    WeightsModule.saveToFile(weights, "$(current_checkpoint_folder)/weight.$(epoch)")
end


