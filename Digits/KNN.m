function [nearest] = KNN(trainSet,trainSetLabel, testSet, k) %Nchunk, chunkSize
    %trainSet=trainv; trainSetLabel=trainlab; testSet=testv; Nchunk=60; chunkSize=1000; k=7;
    
    minDist =      zeros(10000,1);  %zeros(10000, Nchunk);
    minDistIndex = zeros(10000,1);  %zeros(10000, Nchunk);
    %{
    for chunk = 1:Nchunk
        startIterative = (chunk-1)*chunkSize+1;
        endIterative = chunk*chunkSize+1;
        W = trainSet(startIterative:endIterative, :);
        P = testSet.';
        distances = dist(W,P);

        for i=1:size(testSet)
            [sortedVals, indexes] = sort(distances(:,i));
            indexLabel = trainSetLabel(indexes(1:k)+(chunk-1)*chunkSize+1);
            indexLabel = [1;1;2;2;3;4;5];
            [~,~,modeVec] = mode(indexLabel, 'all'); %modeVec is a vector with the numbers are the modes (typetall)
            x = ismember(indexLabel, modeVec{1}); %gives ex. [01011]  if the 2. index matches and so on
            idx = find(x~=0,1,'first'); %gives index of first non zero in x, this is then the first value in
            %modeVec that is then the first value in idexes that is
            %also in modeVec
            minDist(i,chunk) = sortedVals(idx);
            minDistIndex(i,chunk) = indexes(idx)+(chunk-1)*NsamplesOneChunk;
        end
    end
    %}

    W = trainSet;%trainSet(1:100,:); %kjører for bare de 60 første punkene siden det tar for lang tid ellers, hehe
    P = testSet.';

    distances = dist(W,P);

    for i=1:size(testSet)
        [sortedVals, indexes] = sort(distances(:,i)); %sorts by distance
        indexLabel = trainSetLabel(indexes(1:k)); %finds the numbers related to the sorted indexes
        [~,~,modeVec] = mode(indexLabel, 'all'); %modeVec is a vector with the modes of the 7 nearest numbers(typetall)
        firstIndex = ismember(indexLabel, modeVec{1}); %Indicates the number with the shortest distance that mathes 
        %with the vector of modes eg. [0101100]  if the 2. index matches.
        idx = find(firstIndex~=0,1,'first'); %gives index of first non-zero in firstIndex
        minDist(i) = sortedVals(idx); %vector of shortest distance of each data point
        minDistIndex(i) = indexes(idx); %vector of index of shortest distance of each data point
    end

    nearest = trainSetLabel(minDistIndex); %vector of numbers chosen for each data point

end