####
#   Paweł Podgórski
#   
#   
####

push!(LOAD_PATH, "$(pwd())/../common/datasets")
import DigitMNIST
import FashionMNIST
import Iris

include("modules/graph.jl")
include("modules/network.jl")

############### CONFIGURATION
dataset = FashionMNIST
epochs = 1
lr = 0.4
###############

println("Net started!")
### Network module

datasetBase = dataset.createDatasetBase()
train_data = datasetBase.train
test_data = datasetBase.test
classes = datasetBase.classes

println("Use $(dataset.getName()) dataset")
println("train dataset size: $(size(train_data))")
println("test dataset size: $(size(test_data))")

first_layer_width = size(train_data[1][1])[1]
out_layer_width = size(classes)[1]

Wh1 = Variable(randn(32, first_layer_width), name="Wh1")
bh1 = Variable(randn(32), name="bh1")
Wh2 = Variable(randn(32,32), name="Wh2")
bh2 = Variable(randn(32), name="bh2")
Wo = Variable(randn(out_layer_width,32), name="Wo")
bo = Variable(randn(out_layer_width), name="bo")

loss_list = []

for epoch in 1:epochs
    loss = 0.0
    for i in train_data
        x = Constant(i[1])
        y = Constant(i[2])
        train!(x, Wh1, bh1, Wh2, bh2, Wo, bo, y, lr, loss)
        push!(loss_list, loss)
    end
    #println("Epoch ", epoch, " Loss is: ", loss, "\n")
end

println("Result:")
println("Dokładność (test): ", accuracy(test_data))
println("Dokładność (train): ", accuracy(train_data))