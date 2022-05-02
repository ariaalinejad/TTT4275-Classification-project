function [nearest, IndexTrainSet] = NN(trainSet,trainSetLabel, testSet, Nchunk, chunkSize)

    NtestVal = size(testSet,1); 

    minDist = zeros(10000, Nchunk);
    minDistIndex = zeros(10000, Nchunk);
    IndexTrainSet = zeros(10000,1);

    %Find nearest in each chunk
    for chunk = 1 : Nchunk
            startIterative = (chunk-1)*chunkSize+1;
            endIterative = chunk*chunkSize;
            %W = trainSet(startIterative:endIterative,:); 
            %P = testSet.'; 
          
            distances = dist(trainSet(startIterative:endIterative,:), testSet.');
    
            for i = 1 : NtestVal
                [minVal,index] = min(distances(:,i));
                minDist(i,chunk) = minVal;                         %Saving minimum distance for each testvalue
                minDistIndex(i,chunk) = index+(chunk-1)*chunkSize; %Saving index of minimum distance for each testvalue
            end
    end
    
    % Find nearest of all chunks
    for i = 1 : NtestVal
        [~,index] = min(minDist(i,:));
        IndexTrainSet(i) = minDistIndex(i, index);
    end
    
    nearest = trainSetLabel(IndexTrainSet);
end