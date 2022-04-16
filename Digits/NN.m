function [nearest] = NN(trainSet,trainSetLabel, testSet, Nchunk, chunkSize)
    %use dist(template,test) where template is 1000 train samples from templates from test v. 
    %test is 1000 test values, check which has the min length with
    %dist(template,test) and save which indces is the closest to each
    %testvalue.

    NtestVal = size(testSet,1); %Number of testvalues

    minDist = zeros(10000, Nchunk);
    minDistIndex = zeros(10000, Nchunk);
    IndexTrain = zeros(10000,1);

    for chunk = 1 : Nchunk
            startIterative = (chunk-1)*chunkSize+1;
            endIterative = chunk*chunkSize;
            W = trainSet(startIterative:endIterative,:); %Want to use the 1000 values in each chunk as W in dist
            P = testSet.'; % transpose since want 784 rows
            %https://se.mathworks.com/help/deeplearning/ref/dist.html
            distances = dist(W,P);
            %Distances is now a matrix (1000x10 000)of all distances between
            %1000 trainvalues and one testvalue

            %Must check which element in each column is the minimum--> because it
            %is the nearest neighbour.
            %Save this value to a vector minDist
            %Save its iterative to a vecotr minIndicise
            for i = 1 : NtestVal
                [M,I] = min(distances(:,i)); %min distance in each column
                minDist(i,chunk) = M; %inserting min value into minDist matrix 
                minDistIndex(i,chunk) = I+(chunk-1)*chunkSize; %Saving the index to the min value
            end

    end


        %sjekk distanse mellom testverdi i og alle elementene i chunk tail
    for i = 1 : 10000
        %finn ut hvilken som er minst. fra minDist ved bruk av min
        [~,Index] = min(minDist(i,:));%MinVal skal ikke brukes til noex
        %IndexTrain(i) = Index; %Saves the index to IndexTrain. Her skjer det FEIL
        IndexTrain(i) = minDistIndex(i, Index); %Dette tror jeg skal gå.
        % Denne kan brukes til å sjekke hvilket tall som den er funnet til å være.
    end
    
    nearest = trainSetLabel(IndexTrain);
end