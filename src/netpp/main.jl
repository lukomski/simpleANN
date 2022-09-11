####
#   Paweł Podgórski
#   
#   
####

push!(LOAD_PATH, "$(pwd())/../common/datasets")
import DigitMNIST
import FashionMNIST
import Iris

include("$(pwd())/../common/MetricsModule1.jl")

include("modules/graph.jl")
include("modules/network.jl")

############### CONFIGURATION
dataset = Iris
epochs = 5
lr = 0.4
tests = 10
###############

println("Net started!")
### Network module

datasetBase = dataset.createDatasetBase()
train_data = datasetBase.train
test_data = datasetBase.test
classes = datasetBase.classes

create_config_file()

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
loss = 0.0

for test in 1:tests
    save_test_dump(train_data, "train", test, 0)
    save_test_dump(test_data, "test", test, 0)
    for epoch in 1:epochs
        for i in train_data
            x = Constant(i[1])
            y = Constant(i[2])
            train!(x, Wh1, bh1, Wh2, bh2, Wo, bo, y, lr, loss)
            save_test_dump(train_data, "train", test, epoch)
            save_test_dump(test_data, "test", test, epoch)
        end
    end
    println("Test: ", test)
    metrics(test_data, out_layer_width)
end