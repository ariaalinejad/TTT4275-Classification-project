nearest = NN(trainv,trainlab, testv, 60, 100);

title = 'Digit Classification Using the Euclidian distance';
errorRate = confMatrix(nearest, testlab, title);

for i = 1:5
    %%Shit for plotting pics:
    if testlab(i) ~= nearest(i)
        %str = sprintf('MISCLASSIFIED! True digit: %d, Predicted digit: %d',testlab(i),nearest(i));
        imagesc(reshape(testv(i,:),28,28)');
        %title(str);%title("MISCLASSIFIED! True digit: " + testlab(i)+ " Predicted digit: " + nearest(i));
        drawnow;
        pause(1);
        %hold on
    end
    if testlab(i) == nearest(i)
        imagesc(reshape(testv(i,:),28,28)');
        %title( "CORRECLY CLASSIFIED: " + testlab(i) + " ");
        drawnow;
        pause(1);
        %hold on
    end
end
%hold off
        
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

%}