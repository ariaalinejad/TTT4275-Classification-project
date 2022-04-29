 cmTrain = ConfmatrixTrain;
 cmTest = ConfmatrixTest;

 figure(1);
 cm1 = confusionchart(cmTrain, {'Setosa','Versicolor','Verginica'});
 cm1.Title = 'Iris Classification Training ';
 cm1.RowSummary = 'row-normalized';%Displays the error rate
 cm1.ColumnSummary = 'column-normalized';
 cm2.DiagonalColor = '#77AC30';


figure(2);
 cm2 = confusionchart(cmTest, {'Setosa','Versicolor','Verginica'});
 cm2.Title = 'Iris Classification Testing';
 cm2.RowSummary = 'row-normalized';%Displays the error rate
 cm2.ColumnSummary = 'column-normalized';
 cm2.DiagonalColor = '#77AC30';

