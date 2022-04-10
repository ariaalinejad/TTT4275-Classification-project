colormap gray;
%imagesc(reshape(testv(2,:),28,28)');

%trainv = [trainlab,trainv]; %- Denne linjen la til testlab som første kolonne
%i testv, men trengte tydligvis bare kjøre den en gang hehe

%deler matrisen etter verdien på første rad, C blir da en vektor med linker
%til hver av del av matrisen etter verdien på første rad, fra her:
%https://se.mathworks.com/matlabcentral/answers/345111-splitting-matrix-based-on-a-value-in-one-column

[~,~,X] = unique(trainv(:,1));
C = accumarray(X,1:size(trainv,1),[],@(r){trainv(r,:)});

%imagesc(reshape(C{5}(1,2:785),28,28)'); % plotter et tall som er kategorisert

mu = zeros(10,784);
for i = 1:10
    mu(i,:) = sum(C{i}(:,2:785))/length(C{i}(:,1)); %finner summen av en colonne og deler på lengden
    imagesc(reshape(mu(3,:),28,28)'); %printer et gjennomsnittlig tall
end

d = zeros(length(testv(:,1)), length(mu(:,1)));
for i = 1:length(testv(:,1)) %ittererer for hvert sample
    for k = 1:length(mu(:,1)) %ittererer for hver pixel
        d(i,k) = sum(diag((testv(i,2:785) - mu(k,:))'*(testv(i,2:785) - mu(k,:)))); %lager en matriske av hver av avstandene
    end
end

%her lagers det en 784*784 matrise i følge kompendie, men jeg tenker at jeg
%vil lage en 10000*10 matrise som viser avstanden for hvert tall

%her er får jeg ikke testa så mye siden det tar for lang tid

for i = 1:length(d(:,1))
    [minDistanse, minIndex] = min(d(i,:)); % her prøver jeg å finne indeksen til det tallet med minst avstand
end
disp(size(minDistanse))

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