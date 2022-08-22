push!(LOAD_PATH, "$(pwd())/modules")
push!(LOAD_PATH, "$(pwd())/../common/datasets")

# datasets
import DigitMNIST
import FashionMNIST
import Iris

# network
import NetworkModule
import WeightsModule

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
defaultWeightsFolder = "checkpoints"
if (!isdir(defaultWeightsFolder))
    mkdir(defaultWeightsFolder)
end
cd(defaultWeightsFolder)

# save initial weight
WeightsModule.saveToFile(initialWeights, "weight.0")

# load initial weight
weights = WeightsModule.loadFromFile("weight.0")
loadedWeights = weights.weights

epochs = 1
# train one epoch
NetworkModule.train(loadedWeights, train, test, epochs)

# save checkpoint after first epoch
WeightsModule.saveToFile(loadedWeights, "weight.$(epochs)")

println("resut: $(NetworkModule.getStringOfSuccessPercentage(test, loadedWeights))")