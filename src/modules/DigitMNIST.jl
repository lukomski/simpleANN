module DigitMNIST
    using MLDatasets

    function _getPreparedDataset(dataX, dataY)
        collectionSize = size(dataX)[3]
        inputWidth = size(dataX)[1] * size(dataX)[2]

        X = []; 
        Y = []; 

        for i = 1 : collectionSize
            push!(X, reshape(dataX[:,:,i], inputWidth));
            y = zeros(10);
            y[dataY[i] + 1] = 1.0; 
            push!(Y,y);
        end

        outData = [x for x in zip(X,Y)]; 
    end

    function getTestDataset()
        test_x, test_y = MNIST.testdata(Float64);
        _getPreparedDataset(test_x, test_y) 
    end

    function getTrainDataset()
        train_x, train_y = MNIST.traindata(Float64);
        _getPreparedDataset(train_x, train_y) 
    end
end