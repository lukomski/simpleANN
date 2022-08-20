module FashionMNIST
import MLDatasets
import DatasetCommon

function getDataset()
    return MLDatasets.FashionMNIST
end

function getName()
    return "FashionMNIST"
end

function getTestDataset()
    DatasetCommon.getTestDataset(getDataset())
end

function getTrainDataset()
    DatasetCommon.getTrainDataset(getDataset())
end
end
