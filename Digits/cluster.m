M=64;
Ntrain = 6000;

trainlabv = [trainlab,trainv]; 
[~,~,X] = unique(trainlabv(:,1));
trainvSplit = accumarray(X,1:size(trainlabv,1),[],@(r){trainlabv(r,:)});

C = zeros(10,64,784);
%idx = zeros(5923,10);
for i=1:10
    [idx, C(i,:,:)] = kmeans(trainvSplit{i}(:,2:785),M);
end

