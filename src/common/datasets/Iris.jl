module Iris
import MLDatasets
import DatasetCommon
using Shuffle

function getDataset()
    return MLDatasets.Iris
end

function getName()
    return "Iris"
end

function getClasses()
    classes = sort(unique(MLDatasets.Iris.labels()))
    return classes
end

function getPreparedDataset(dataX, dataY)
    collectionSize = size(dataX)[2]
    inputWidth = size(dataX)[1]

    classes = getClasses()

    X = []
    Y = []

    for i = 1:collectionSize
        push!(X, reshape(dataX[:, i], inputWidth))
        y = zeros(size(classes)[1])
        class_idx = findfirst(isequal(dataY[i]), classes)
        if (class_idx === nothing)
            println("Unkown class $(dataY[i])")
        end
        y[Int128(class_idx)] = 1.0
        push!(Y, y)
    end

    outData = [x for x in zip(X, Y)]
end

function createDatasetBase()
    features = MLDatasets.Iris.features()
    labels = MLDatasets.Iris.labels()

    len = size(features)[2]
    # make table of shuffle
    shuffle_table = zeros(Int128, 2, len)
    for idx = 1:len
        shuffle_table[:, idx] = [idx, idx]
    end
    shuffle_table[2, :] = shuffle(shuffle_table[2, :])

    new_order_features = zeros(size(features))
    new_order_labels = String[]

    for idx = 1:len
        new_idx = shuffle_table[2, idx]
        new_order_features[:, idx] = features[:, new_idx]
        push!(new_order_labels, labels[new_idx])
    end


    percent_of_test_dataset = 15
    quantity_of_test_dataset = floor(Int128, len * (percent_of_test_dataset / 100))
    quantity_of_train_dataset = len - quantity_of_test_dataset

    test_features = new_order_features[:, 1:quantity_of_test_dataset]
    test_labels = new_order_labels[1:quantity_of_test_dataset]

    train_features = new_order_features[:, quantity_of_test_dataset+1:len]
    train_labels = new_order_labels[quantity_of_test_dataset+1:len]

    return DatasetCommon.DatasetBase(
        getPreparedDataset(train_features, train_labels),
        getPreparedDataset(test_features, test_labels),
        getClasses()
    )

end

function download()
    MLDatasets.Iris.download(undef, true)
end

end
