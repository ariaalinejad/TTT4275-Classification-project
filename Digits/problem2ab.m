M=64;
Nnums = 10;
NColsTrain = size(trainv,2);

trainlabv = [trainlab,trainv]; 
[~,~,X] = unique(trainlabv(:,1));
trainvSplit = accumarray(X,1:size(trainlabv,1),[],@(r){trainlabv(r,:)});

C = zeros(Nnums*M,NColsTrain);
C_labels = zeros(Nnums*M);
for i=1:10
    [idx, C((i-1)*M+1:i*M,:)] = kmeans(trainvSplit{i}(:,2:785),M);
    C_labels((i-1)*M+1:i*M) = i-1;
end

nearest = NN(C,C_labels,testv,Nnums,M);

title = 'Digit Classification Using the Kmeans';
errorRate = confMatrix(nearest, testlab, title);



