
test = KNN(trainv,testv,10,64,1);

function [label] = KNN(trainSet, testSet, Nchunk, chunkSize, k)
    minDist = zeros(10000, Nchunk);
    minDistIndex = zeros(10000, Nchunk);
    
    for chunk = 1:Nchunk
        startIterative = (chunk-1)*chunkSize+1;
        endIterative = chunk*chunkSize+1;
        W = trainSet(startIterative:endIterative, :);
        P = testSet.';
        distances = dist(W,P);
        
        for i=1:size(testSet)
            [M,I] = min(distances(:,i));
            minDist(i,chunk) = M;
            minDistIndex(i,chunk) = I+(chunk-1)*NsamplesOneChunk;
        end
    end

    label = 5; %gir ut

end