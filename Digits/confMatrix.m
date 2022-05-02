function [errorRate] = confMatrix(classifiedDigit, testLabel, title)
   
    confmatrix = zeros(10,10);
    error = 0;

    for i = 1: 10000
        confmatrix(testLabel(i)+1,classifiedDigit(i)+1) = confmatrix(testLabel(i)+1,classifiedDigit(i)+1)+1;
   
        if testLabel(i) ~= classifiedDigit(i)
            error = error + 1;
        end
    end

    % Plotting confusion matrix and finding error rate
    cm = confusionchart(confmatrix,{'0','1','2','3','4','5','6','7','8','9'});
    cm.Title = title; 
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    cm.DiagonalColor = '#77AC30';

    errorRate = error/10000;
end