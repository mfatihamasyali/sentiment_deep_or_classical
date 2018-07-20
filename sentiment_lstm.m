% (c) Mehmet Fatih Amasyali, 2018
clear all;
close all;
rng(1);
% sentiment analysis (actually text classification, you can change train and test files for your text classification application)

% Method: lstm with word embeddings obtained from training files 

% Train and test files are in excel format. 
% first column: tweet text 
% second column: tweet label (positive, negative, notr)
trainfile='train_tweets.xlsx'; 
testfile='test_tweets.xlsx';
min_fre=3; % min_frequence of words
embeddingDimension = 300; % dimension of word embedding 
embeddingEpochs = 100; % epoch number for training word embedding 
labels=[ "olumsuz", "notr", "olumlu"];
numClasses = size(labels,2); 
num_hidden = 30; % number of hidden units in LSTM 
maxEpochs=30; % epoch for LSTM training
maxLength = 30; % max word count in a tweet, for longer tweets the first maxLength words are keep

train_table=readtable(trainfile,'ReadVariableNames',false);
test_table=readtable(testfile,'ReadVariableNames',false);
train_tweets=string(table2cell(train_table(:,1)));
test_tweets=string(table2cell(test_table(:,1)));
train_labels=categorical(table2cell(train_table(:,2)));
test_labels=categorical(table2cell(test_table(:,2)));

train_docs = tokenizedDocument(train_tweets);
train_docs = lower(train_docs); % convert tweets to lowercase
test_docs = tokenizedDocument(test_tweets);
test_docs = lower(test_docs); 

% train word embedding with training files 
documentsTrain=train_docs;
tic
tw_emb = trainWordEmbedding(train_docs, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'NGramRange',[0 0], ... % do not use subword similarity
    'MinCount',min_fre, ... 
    'Verbose',0);
toc
fast_emb=tw_emb;

documentsTruncatedTrain = docfun(@(words) words(1:min(maxLength,end)),documentsTrain);
XTrain = doc2sequence(fast_emb,documentsTruncatedTrain);
for i = 1:numel(XTrain)
    XTrain{i} = leftPad(XTrain{i},maxLength);
end

inputSize = embeddingDimension;
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(num_hidden,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'Plots','training-progress', ...
    'MaxEpochs',maxEpochs, ...
    'Verbose',0);

net = trainNetwork(XTrain,train_labels,layers,options);

textDataTest=test_docs;
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);
documentsTruncatedTest = docfun(@(words) words(1:min(maxLength,end)),documentsTest);
XTest = doc2sequence(fast_emb,documentsTruncatedTest);
for i=1:numel(XTest)
    XTest{i} = leftPad(XTest{i},maxLength);
end
YPred = classify(net,XTest);
accuracy = sum(YPred == test_labels)/numel(YPred)

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
