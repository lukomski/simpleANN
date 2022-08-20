module Iris
import MLDatasets
import DatasetCommon

function getDataset()
    return MLDatasets.Iris
end

function getName()
    return "Iris"
end

# function getTestDataset()
#     DatasetCommon.getTestDataset(getDataset())
# end

# function getTrainDataset()
#     DatasetCommon.getTrainDataset(getDataset())
# end

function download()
    MLDatasets.Iris.download("./iris_dataset")
end

function features()
    MLDatasets.Iris.features()
end

end
