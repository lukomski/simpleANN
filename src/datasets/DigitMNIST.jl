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
end