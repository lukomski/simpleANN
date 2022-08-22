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

function getClasses()
    classes = sort(unique(MLDatasets.FashionMNIST.testdata(Float64)[2]))
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
