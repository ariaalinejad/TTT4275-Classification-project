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
for i=1:10
    [idx, C((i-1)*M+1:i*M,:)] = kmeans(trainvSplit{i}(:,2:785),M);
    C_labels((i-1)*M+1:i*M) = i-1;
end

%% Classify digits using KNN (K = 3) with clustering, plot confusion matrix
KNNclustNearest = KNN(C, C_labels, testv, 7);

cmKNNtitle = 'KNN digit Classification Using clustering';
errorRateKNNclust = confMatrix(KNNclustNearest, testlab, cmKNNtitle);

toc

%% Plotting centroid of digit 2 and 5
Nsamples = length(trainvSplit{6});
NmeanSamples = round(Nsamples/M);
sumSamples2=sum(trainvSplit{3}(1:NmeanSamples,:),1);
meanSamples2 = sumSamples2/NmeanSamples;
sumSamples5=sum(trainvSplit{6}(NmeanSamples:2*NmeanSamples,:),1);
meanSamples5 = sumSamples5/NmeanSamples;

colormap(gray);
figure(1);
subplot(1,2,1);
imagesc(reshape(meanSamples2(:,2:785),28,28)');
title('Sample of centroid of digit 2');
subplot(1,2,2);
imagesc(reshape(meanSamples5(:,2:785),28,28)');
title('Sample of centroid of digit 5');






