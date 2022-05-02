%Finds the nearest neighbor with clustering
tic
M=64;
Nnums = 10;
NColsTrain = size(trainv,2);

%% Split data into classes
trainlabv = [trainlab,trainv]; 
[~,~,X] = unique(trainlabv(:,1)); 
trainvSplit = accumarray(X,1:size(trainlabv,1),[],@(r){trainlabv(r,:)});

%% Clustering
C = zeros(Nnums*M,NColsTrain);
C_labels = zeros(Nnums*M);
for i=1:10%itererere gjennom alle cellene
    [~, C((i-1)*M+1:i*M,:)] = kmeans(trainvSplit{i}(:,2:785),M); 
    C_labels((i-1)*M+1:i*M) = i-1;
end

%% Classify digits using NN with clustering, plot confusion matrix
nearestNNclust = NN(C,C_labels,testv,1,M*Nnums);

cmNNclustTitle = 'NN digit Classification using clustering';
errorRateNNclust = confMatrix(nearestNNclust, testlab, cmNNclustTitle)


toc



