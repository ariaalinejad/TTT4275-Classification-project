colormap gray;


% %use dist(template,test) where template is 1000 train samples from templates from test v. 
% %test is 1000 test values, check which has the min length with
% %dist(template,test) and save which indces is the closest to each
% %testvalue.
% NsamplesOneChunk = 1000; %Number of samples in one chunk
% Nchunk = size(trainv,1)/NsamplesOneChunk; %Number of chunks that have to be compared to each test value
% NtestVal = size(testv,1); %Number of testvalues
% 
% minDist = zeros(10000, Nchunk);
% minDistIndex = zeros(10000, Nchunk);
% IndexTrain = zeros(10000,1);
% 
% for chunk = 1 : Nchunk
%         startIterative = (chunk-1)*NsamplesOneChunk+1;
%         endIterative = chunk*NsamplesOneChunk
%         W = trainv(startIterative:endIterative,:); %Want to use the 1000 values in each chunk as W in dist
%         P = testv.'; % transpose since want 784 rows
%         %https://se.mathworks.com/help/deeplearning/ref/dist.html
%         distances = dist(W,P);
%         %Distances is now a matrix (1000x10 000)of all distances between
%         %1000 trainvalues and one testvalue
% 
%         %Must check which element in each column is the minimum--> because it
%         %is the nearest neighbour.
%         %Save this value to a vector minDist
%         %Save its iterative to a vecotr minIndicise
%         for i = 1 : NtestVal
%             [M,I] = min(distances(:,i)); %min distance in each column
%             minDist(i,chunk) = M; %inserting min value into minDist matrix 
%             minDistIndex(i,chunk) = I+(chunk-1)*NsamplesOneChunk; %Saving the index to the min value
%         end
% 
% end
% 
% 
%     %sjekk distanse mellom testverdi i og alle elementene i chunk tail
% for i = 1 : 10000
%     %finn ut hvilken som er minst. fra minDist ved bruk av min
%     [~,Index] = min(minDist(i,:));%MinVal skal ikke brukes til noe
%     %IndexTrain(i) = Index; %Saves the index to IndexTrain. Her skjer det FEIL
%     IndexTrain(i) = minDistIndex(i, Index); %Dette tror jeg skal gå.
%     % Denne kan brukes til å sjekke hvilket tall som den er funnet til å være.
% end
% Det eneste jeg lurer på er hvorfor oppgaven sier avoid using
%excessive time (as when classifying a single image at a time) for det gjør
%jo vi nå...

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
    confmatrix(testlab(i)+1,trainlab(IndexTrain(i))+1) = confmatrix(testlab(i)+1,trainlab(IndexTrain(i))+1)+1;
    %%Shit for plotting pics:
    if testlab(i) == 4 && trainlab(IndexTrain(i))== 4
        imagesc(reshape(testv(i,:),28,28)');
        title("true digit: 4, predicted digit: 4. Index test:", i)
    end

    %%To find errorRate
    if testlab(i) ~= trainlab(IndexTrain(i))
        error = error + 1;
    end
end
%%Plotting confusion matrix and finding error rate
 cm = confusionchart(confmatrix);
 cm.Title = 'Digit Classification Using the Euclidian distance';
 cm.RowSummary = 'row-normalized';%Displays the error rate
 cm.ColumnSummary = 'column-normalized';


 errorRate = error/10000;



