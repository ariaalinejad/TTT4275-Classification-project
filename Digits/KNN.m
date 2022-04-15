

trainSet=trainv; trainSetLabel=trainlab; testSet=testv; Nchunk=60; chunkSize=1000; k=7;

%function [label] = knear(trainSet,trainSetLabel, testSet, Nchunk, chunkSize, k)
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

W = trainSet(1:1000,:); %kjører for bare de 60 første punkene siden det tar for lang tid ellers, hehe
P = testSet.';

distances = dist(W,P);

for i=1:size(testSet)
    [sortedVals, indexes] = sort(distances(:,i));
    indexLabel = trainSetLabel(indexes(1:k));
    [~,~,modeVec] = mode(indexLabel, 'all'); %modeVec is a vector with the numbers are the modes (typetall)
    x = ismember(indexLabel, modeVec{1}); %gives ex. [01011]  if the 2. index matches and so on
    idx = find(x~=0,1,'first'); %gives index of first non zero in x, this is then the first value in
    %modeVec that is then the first value in idexes that is
    %also in modeVec
    minDist(i) = sortedVals(idx);
    minDistIndex(i) = indexes(idx);
end

nearestNeighbors = trainSetLabel(minDistIndex);
%Nå kan jeg gi ut indexen til det tallet som har lavest avstand, wihoooo

title = 'Digit Classification Using the Kmeans';
errorRate = confMatrix(nearestNeighbors, testlab, title);

%label = 5; %gir ut

%end