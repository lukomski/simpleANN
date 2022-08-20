push!(LOAD_PATH, "$(pwd())/modules")
push!(LOAD_PATH, "$(pwd())/datasets")

# datasets
import DigitMNIST
import FashionMNIST

# network
import Network

# configure dataset
dataset = Iris

# display selected confiration
println("Use $(dataset.getName()) dataset")

train = dataset.getTrainDataset()
test = dataset.getTestDataset()

println("train dataset size: $(size(train))")
println("test dataset size: $(size(test))")

weights = Network.Weights()
epochs = 1
new_weights = Network.train(weights, train, test, epochs)

println("resut: $(Network.getStringOfSuccessPercentage(test, weights))")