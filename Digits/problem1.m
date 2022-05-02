%%Finds the nearest neighbor without clustering

tic

%% Classify digits using NN, plot confusion matrix

[nearestNN, indexTrain] = NN(trainv,trainlab, testv, 60, 1000);

figure(1);
cmTitle = 'Digit Classification Using the Euclidian distance';
errorRateNN = confMatrix(nearestNN, testlab, cmTitle)

%% Find correctly and misclassified digits
CorrectClassifiedIndex = [];
misclassifiedIndex = [];
indexTrainMiss = [];


for i = 1:10000
    if testlab(i) == nearestNN(i)
        [~, correctIndex] = unique(testlab);
    end
end


for i = 1:10000
    if testlab(i) ~= nearestNN(i)
        misclassifiedIndex(end+1) = i;
        indexTrainMiss(end+1) = indexTrain(i);
    end
end

%% Plotting correct classified
colormap(gray)
figure(2);
sgtitle("NN Correctly classified digits");
for i = 1:10
    subplot(2,5,i);
    imagesc(reshape(testv(correctIndex(i),:),28,28)');
    title(['Predicted digit: ', num2str(nearestNN(correctIndex(i))),', True digit: ',num2str(testlab(correctIndex(i)))]);
end

%% Plotting misclassified
colormap(gray)
figure(3);
sgtitle("NN Misclassified digits");
for i = 1:2:8
    subplot(4,2,i);
    imagesc(reshape(testv(misclassifiedIndex(i),:),28,28)');
    title(['Predicted digit: ', num2str(nearestNN(misclassifiedIndex(i))), ', True digit: ',num2str(testlab(misclassifiedIndex(i)))]);

    subplot(4,2,i+1);
    imagesc(reshape(trainv(indexTrainMiss(i),:),28,28)');
    title('The nearest neighbor ');
end

toc





