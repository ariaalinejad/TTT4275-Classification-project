nearest = KNN(trainv, trainlab, testv,7);

title = 'Digit Classification Using the Kmeans';
errorRate = confMatrix(nearest, testlab, title);