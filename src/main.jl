push!(LOAD_PATH, "$(pwd())/modules")

import DigitMNIST
import Network

train = DigitMNIST.getTrainDataset()
test = DigitMNIST.getTestDataset()

weights = Network.Weights()
txt = Network.success_percentage(test, weights)
println(txt)

for i=1:size(train)[1]
    x = Network.Constant(train[i][1])
    y = Network.Constant(train[i][2])
    Network.update_weights!(x,weights, y)
end
txt = Network.success_percentage(test, weights)
println(txt)