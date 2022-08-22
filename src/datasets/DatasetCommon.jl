module DatasetCommon
function getPreparedDataset(dataX, dataY)
    collectionSize = size(dataX)[3]
    inputWidth = size(dataX)[1] * size(dataX)[2]

    X = []
    Y = []

    for i = 1:collectionSize
        push!(X, reshape(dataX[:, :, i], inputWidth))
        y = zeros(10)
        y[dataY[i]+1] = 1.0
        push!(Y, y)
    end

    outData = [x for x in zip(X, Y)]
end

function getTestDataset(dataset)
    test_x, test_y = dataset.testdata(Float64)
    DatasetCommon.getPreparedDataset(test_x, test_y)
end

function getTrainDataset(dataset)
    train_x, train_y = dataset.traindata(Float64)
    DatasetCommon.getPreparedDataset(train_x, train_y)
end

mutable struct DatasetBase
    classes::Any
    test::Any
    train::Any
    DatasetBase(train, test, classes) = new(classes, test, train)
end
end