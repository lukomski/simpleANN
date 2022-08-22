push!(LOAD_PATH, "$(pwd())/modules")
push!(LOAD_PATH, "$(pwd())/datasets")

# datasets
import DigitMNIST
import FashionMNIST
import Iris

# network
import Network

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

firstLayerWidth = size(train[1][1])[1]
outLayerWidth = size(classes)[1]

weights = Network.Weights(firstLayerWidth, outLayerWidth)
epochs = 1
Network.train(weights, train, test, epochs)

println("resut: $(Network.getStringOfSuccessPercentage(test, weights))")