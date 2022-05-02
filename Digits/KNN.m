function [nearest] = KNN(trainSet,trainSetLabel, testSet, k) 
     
    minDistIndex = zeros(10000,1); 
  
    distances = dist(trainSet,testSet.');               %Compute Euclidean distances

    for i=1:size(testSet)
        [~ , indexes] = sort(distances(:,i)); 
        indexLabel = trainSetLabel(indexes(1:k));      %Finds the digits related to the sorted indexes
        [~,~,modeVec] = mode(indexLabel, 'all');       %modeVec is a vector of the indexes of the mode(s) 
        firstIndex = ismember(indexLabel, modeVec{1}); %Find what digit of the mode(s) is nearest
        idx = find(firstIndex~=0, 1, 'first');         %idx is 1,2,..,7 if digit is nearest, next nearest,.., 7'th nearest
        minDistIndex(i) = indexes(idx);                %Save index of the digit chosen by KNN 
      
    end

    nearest = trainSetLabel(minDistIndex); 

end