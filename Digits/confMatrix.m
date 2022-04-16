function [errorRate] = confMatrix(trainLabel, testLabel, title)
    %%Find the confusion matrix and the error rate for the test set
    %use confusionchart(trueLabels,predictedLabels) for confusion matrix
    %Jeg har alle indexene til testverdiene i lista IndexTrain. Disse har
    %symbol lik trainlab(index) og egentlige verdi lik testlab(i). Sett alle
    %disse verdiene inn i en matrise som skal bli brukt i condusion matrix.

    confmatrix = zeros(10,10);
    % PredictedVal = zeros(10000,1);
    error = 0;

    for i = 1: 10000
        %+1 siden ikke nullindeksert og et av siferne er 0
        confmatrix(testLabel(i)+1,trainLabel(i)+1) = confmatrix(testLabel(i)+1,trainLabel(i)+1)+1;
   
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