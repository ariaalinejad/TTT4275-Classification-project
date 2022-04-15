function [errorRate] = confMatrix(trainLabel, testLabel, title)
    confmatrix = zeros(10,10);
    % PredictedVal = zeros(10000,1);
    error = 0;

    for i = 1: 10000
        %+1 siden ikke nullindeksert og et av siferne er 0
        confmatrix(testLabel(i)+1,trainLabel(i)+1) = confmatrix(testLabel(i)+1,trainLabel(i)+1)+1;
        %%Shit for plotting pics:
        %{
        if testLabel(i) == 4 && trainLabel(i)== 4
            imagesc(reshape(testSet(i,:),28,28)');
            title("true digit: 4, predicted digit: 4. Index test:", i)
        end
        %}
        %%To find errorRate
        if testLabel(i) ~= trainLabel(i)
            error = error + 1;
        end
    end
    %%Plotting confusion matrix and finding error rate
    cm = confusionchart(confmatrix);
    cm.Title = title; %'Digit Classification Using the Euclidian distance';
    cm.RowSummary = 'row-normalized';%Displays the error rate
    cm.ColumnSummary = 'column-normalized';

    errorRate = error/10000;
end