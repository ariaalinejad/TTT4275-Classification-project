
nearest = NN(trainv,trainlab, testv, 60, 100);

cmTitle = 'Digit Classification Using the Euclidian distance';
errorRate = confMatrix(nearest, testlab, cmTitle);



CorrectClassifiedIndex = [];
misclassifiedIndex = [];


for i = 1:10000
    if testlab(i) == nearest(i)
        CorrectClassifiedIndex(end+1) = i;  
    end
end

for i = 1:10000
    if testlab(i) ~= nearest(i)
        misclassifiedIndex(end+1) = i;
    end
end


colormap(gray)
figure(1);
sgtitle("NN Correctly classified digits");
for i = 1:8
    subplot(4,2,i);
    imagesc(reshape(testv(CorrectClassifiedIndex(i),:),28,28)');
    title(['True digit: ',num2str(testlab(CorrectClassifiedIndex(i))), ', Predicted digit: ', num2str(nearest(CorrectClassifiedIndex(i)))]);
end

figure(2);
sgtitle("NN Misclassified digits");
for i = 1:8
    subplot(4,2,i);
    imagesc(reshape(testv(misclassifiedIndex(i),:),28,28)');
    title(['True digit: ',num2str(testlab(misclassifiedIndex(i))), ', Predicted digit: ', num2str(nearest(misclassifiedIndex(i)))]);
end








