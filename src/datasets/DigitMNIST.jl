module DigitMNIST
using MLDatasets
import DatasetCommon

function getDataset()
    return MNIST
end

function getName()
    return "DigitMNIST"
end

function getTestDataset()
    DatasetCommon.getTestDataset(getDataset())
end

function getTrainDataset()
    DatasetCommon.getTrainDataset(getDataset())
end

function getClasses()
    classes = sort(unique(MNIST.testdata(Float64)[2]))
    return classes
end

function createDatasetBase()
    return DatasetCommon.DatasetBase(
        getTrainDataset(),
        getTestDataset(),
        getClasses()
    )
end
end