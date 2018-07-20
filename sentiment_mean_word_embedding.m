% (c) Mehmet Fatih Amasyali, 2018
clear all;
close all;
rng(1); 
% sentiment analysis (actually text classification, you can change train and test files for your text classification application)

% Method: texts are represented with the means of word embeddings trained with training texts.

% Train and test files are in excel format. 
% first column: tweet text 
% second column: tweet label (positive, negative, notr)
trainfile='train_tweets.xlsx'; 
testfile='test_tweets.xlsx';
min_fre=2; % min_frequence of words
embeddingDimension = 300; % dimension of word embedding 
embeddingEpochs = 100; % epoch number for training word embedding 
knn_k=10; % k value of knn
labels=[ "olumsuz", "notr", "olumlu"]; 

train_table=readtable(trainfile,'ReadVariableNames',false);
test_table=readtable(testfile,'ReadVariableNames',false);
train_tweets=string(table2cell(train_table(:,1)));
test_tweets=string(table2cell(test_table(:,1)));
train_labels=string(table2cell(train_table(:,2)));
test_labels=string(table2cell(test_table(:,2)));

train_docs = tokenizedDocument(train_tweets);
train_docs = lower(train_docs); % convert tweets to lowercase
test_docs = tokenizedDocument(test_tweets);
test_docs = lower(test_docs); 

% train word embedding with training files 
tic
tw_emb = trainWordEmbedding(train_docs, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'NGramRange',[0 0], ... % do not use subword similarity
    'MinCount',min_fre, ... 
    'Verbose',0);
toc
fast_emb=tw_emb;

train_bag = bagOfWords(train_docs);
train_voc = train_bag.Vocabulary;
TrM=train_bag.Counts; 
train_voc_vec = word2vec(fast_emb,train_voc);  
vekvar=find(isnan(train_voc_vec(:,1))==0);
train_voc2=train_voc(vekvar); 
train_voc_vec2=train_voc_vec(vekvar,:); % get training word's vectors
TrM=TrM(:,vekvar); % remove words which have not word vectors from traning term document matrix 
disp(['number of unique training file words =' int2str(size(train_voc,2))]);
disp(['number of unique training file words having word vectors =' int2str(size(train_voc2,2))]);

test_bag = bagOfWords(test_docs);
test_voc = test_bag.Vocabulary;
TsM=test_bag.Counts;
test_voc_vec = word2vec(fast_emb,test_voc); 
vekvar=find(isnan(test_voc_vec(:,1))==0);
test_voc2=test_voc(vekvar); 
test_voc_vec2=test_voc_vec(vekvar,:); % get testing file word's vectors
TsM=TsM(:,vekvar); % remove words which have not word vectors from testing term document matrix
disp(['number of unique testing file words =' int2str(size(test_voc,2))]);
disp(['number of unique testing file words having word vectors =' int2str(size(test_voc2,2))]);

% calculate mean word vectors 
% for training files
trainN=size(TrM,1);
trainVec=zeros(trainN,embeddingDimension);
for i=1:trainN
    wv=train_voc_vec2(TrM(i,:)>0,:);
    trainVec(i,:)=sum(wv)/size(wv,1); 
end
% for test files
testN=size(TsM,1);
testVec=zeros(testN,embeddingDimension);
for i=1:testN
    wv=test_voc_vec2(TsM(i,:)>0,:);
    testVec(i,:)=sum(wv)/size(wv,1); 
end

% calculate distances between test and training tweets 
% for each test tweet, find the nearest knn_k training tweet 
[Deg,Ind] = pdist2(trainVec,testVec,'cosine','Smallest',knn_k);
preds=test_labels;
tic
for i=1:testN
    tahmin_k=train_labels(Ind(:,i));% training indices which are nearest to the i.th test tweet 
    pN=size(find(tahmin_k=='olumlu'),1);   nN=size(find(tahmin_k=='olumsuz'),1);    ntN=size(find(tahmin_k=='notr'),1);
    [~, pred_ind]=max([nN ntN pN]);
    preds(i)=labels(pred_ind);
end
toc

acc_knn = sum(preds == test_labels)/testN

%if you use these vectors with other algorithms, uncomment below code
% XTrain=trainVec;
% YTrain=train_labels;
% XTest=testVec;
% YTest=test_labels;
% 
% tree = fitctree(XTrain,YTrain); % Decision Tree
% RF = TreeBagger(100,XTrain,YTrain,'OOBPrediction','On','Method','classification'); % Random Forest
% SVM = fitcecoc(XTrain,YTrain); % SVM
% 
% YPred1 = predict(tree,XTest);
% YPred2 = predict(RF,XTest);
% YPred3 = predict(SVM,XTest);
% 
% acc_tree = sum(YPred1 == YTest)/numel(YTest)
% acc_RF = sum(YPred2 == YTest)/numel(YTest)
% acc_SVM = sum(YPred3 == YTest)/numel(YTest)


% Copyright Mehmet Fatih Amasyali, 2018 All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.




