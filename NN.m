colormap gray;


%use dist(template,test) where template is 1000 train samples from templates from test v. 
%test is 1000 test values, check which has the min length with
%dist(template,test) and save which indces is the closest to each
%testvalue.
NsamplesOneChunk = 1000; %Number of samples in one chunk
Nchunk = size(trainv,1)/NsamplesOneChunk; %Number of chunks that have to be compared to each test value
NtestVal = size(testv,1); %Number of testvalues

minDist = zeros(10000, Nchunk);
minDistIndex = zeros(10000, Nchunk);
IndexTrain = zeros(10000,1);

for chunk = 1 : Nchunk
        startIterative = (chunk-1)*NsamplesOneChunk+1;
        endIterative = chunk*NsamplesOneChunk
        W = trainv(startIterative:endIterative,:); %Want to use the 1000 values in each chunk as W in dist
        P = testv.'; % transpose since want 784 rows
        %https://se.mathworks.com/help/deeplearning/ref/dist.html
        distances = dist(W,P);
        %Distances is now a matrix (1000x10 000)of all distances between
        %1000 trainvalues and one testvalue

        %Must check which element in each column is the minimum--> because it
        %is the nearest neighbour.
        %Save this value to a vector minDist
        %Save its iterative to a vecotr minIndicise
        for i = 1 : NtestVal
            [M,I] = min(distances(:,i)); %min distance in each column
            minDist(i,chunk) = M; %inserting min value into minDist matrix 
            minDistIndex(i,chunk) = I+(chunk-1)*NsamplesOneChunk; %Saving the index to the min value
        end

end


    %sjekk distanse mellom testverdi i og alle elementene i chunk tail
for i = 1 : 10000
    %finn ut hvilken som er minst. fra minDist ved bruk av min
    [MinVal,Index] = min(minDist(i,:));%MinVal skal ikke brukes til noe
    %IndexTrain(i) = Index; %Saves the index to IndexTrain. Her skjer det FEIL
    IndexTrain(i) = minDistIndex(i, Index); %Dette tror jeg skal gå.
    % Denne kan brukes til å sjekke hvilket tall som den er funnet til å være.
end
% Det eneste jeg lurer på er hvorfor oppgaven sier avoid using
%excessive time (as when classifying a single image at a time) for det gjør
%jo vi nå...

%%Find the confusion matrix and the error rate for the test set
%use confusionchart(trueLabels,predictedLabels) for confusion matrix
%Jeg har alle indexene til testverdiene i lista IndexTrain. Disse har
%symbol lik trainlab(index) og egentlige verdi lik testlab(i). Sett alle
%disse verdiene inn i en matrise som skal bli brukt i condusion matrix.
confmatrix = zeros(10,10);

for i = 1: 10000
    %Fuck noe er gAlt. Må ha gjort noe feil enten ved printing av conf,
    %eller ved Euculidian. Tror det er feil i euclidian.
    confmatrix(testlab(i)+1,trainlab(IndexTrain(i))+1) = confmatrix(testlab(i)+1,trainlab(IndexTrain(i))+1)+1;
end

%+1 siden ikke nullindeksert og et av siferne er 0









% %imagesc(reshape(testv(2,:),28,28)');
% 
% trainvNew = [trainlab,trainv]; %- Denne linjen la til testlab som første kolonne
% %i testv, men trengte tydligvis bare kjøre den en gang hehe
% 
% %deler matrisen etter verdien på første rad, C blir da en vektor med linker
% %til hver av del av matrisen etter verdien på første rad, fra her:
% %https://se.mathworks.com/matlabcentral/answers/345111-splitting-matrix-based-on-a-value-in-one-column
% 
% [~,~,X] = unique(trainvNew(:,1));
% C = accumarray(X,1:size(trainvNew,1),[],@(r){trainvNew(r,:)});
% %%accumdata summerer values from Data using sum. den sorterer ikke bare i
% %%gruppe.
% 
% image(reshape(C{3}(118,2:785),28,28)'); % plotter et tall som er kategorisert
% 
% %Likning 16
%  mu = zeros(10,784); %10 siffer, 784 pixls
%  for i = 1:10 %for alle siffer  Vi summere alle de samme pikslene med hverandre
%      mu(i,:) = sum(C{i}(:,2:785),1)/size(C{i},1); %finner summen av en colonne og deler på lengden
%      imagesc(reshape(mu(i,:),28,28)'); %printer et gjennomsnittlig tall
% end
% 
% % mu(1,:) = sum(C{1},1)/size(C{1},1); %finner summen av en colonne og deler på lengden
% imagesc(reshape(mu(5,:),28,28)');

% bruker heller
%dist(template,test) 


% d = zeros(length(testv(:,1)), length(mu(:,1)));
% for i = 1:length(testv(:,1)) %ittererer for hvert sample
%     for k = 1:length(mu(:,1)) %ittererer for hver pixel
%         d(i,k) = sum(diag((testv(i,2:785) - mu(k,:))'*(testv(i,2:785) - mu(k,:)))); %lager en matrise av hver av avstandene
%     end
% end

%her lagers det en 784*784 matrise i følge kompendie, men jeg tenker at jeg
%vil lage en 10000*10 matrise som viser avstanden for hvert tall 
% (10000 testverdier og 10 ulike siffer)

%her er får jeg ikke testa så mye siden det tar for lang tid
% 
% for i = 1:length(d(:,1))
%     [minDistanse, minIndex] = min(d(i,:)); % her prøver jeg å finne indeksen til det tallet med minst avstand
% end
% disp(size(minDistanse))

%Til neste gang: 
%redusere antall samples til 1000 og teste videre med funk-en som er over,
%siden sånn det er nå tar det for lang tid. så må jeg klare å lage en
%vektor over hvilke tall vi klassifiserer det til å være


%Oppgave a)
%del data inn in forskjellige tall, finn gjennomslittet av tallet
%bruk ecu distance til å klassifisere test tall inn i riktig karegori
%du setter den til å være det gjennomsnittet den er nærmes
%De sier at vi skal dele dataen inn i bolker på 1000, men skjønner ikke
%helt hvorfor...


