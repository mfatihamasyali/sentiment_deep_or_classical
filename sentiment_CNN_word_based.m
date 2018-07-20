% (c) Mehmet Fatih Amasyali, 2018
clear all;
close all;
rng(1);
% sentiment analysis (actually text classification, you can change train and test files for your text classification application)

% Method: convert tweets (word based) to images then train with CNNs 

% Train and test files are in excel format. 
% first column: tweet text 
% second column: tweet label (positive, negative, notr)
trainfile='train_tweets.xlsx'; 
testfile='test_tweets.xlsx';
min_fre=3; % min_frequence of words
embeddingDimension = 100; % dimension of word embedding 
embeddingEpochs = 50;
maxLength=40; % max word count in a tweet, for longer tweets the first maxLength words are keep

cnn_filter_width=3; % width of cnn filters
cnn_filter_N=200; % number of cnn filters
maxEpochs=30; % epoch for CNN training
drop_prob=0.66; % dropout prob.
labels=[ "olumsuz", "notr", "olumlu"];
numClasses = size(labels,2); 

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

train_N=size(train_labels,1);
test_N=size(test_labels,1);

image_row_size=maxLength;
image_col_size=embeddingDimension;

tic
tw_emb = trainWordEmbedding(train_docs, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'MinCount',min_fre, ... 
    'NGramRange',[0 0], ...
    'Verbose',0);
toc

myemb=tw_emb;
Voc=myemb.Vocabulary;
dins=word2vec(myemb,Voc);
minD=min(min(dins));
maxD=max(max(dins)); % get min and max values of word vectors
TRAIN_TW=zeros(image_row_size,image_col_size,1,train_N);
TEST_TW=zeros(image_row_size,image_col_size,1,test_N);
tic
for i=1:train_N
    raw = word2vec(myemb,string(train_docs(i)));
    var_i=find(isnan(raw(:,1))==0);
    raw=raw(var_i,:); % words having wordembedding 
    raw=raw+abs(minD); % add min. the values are between 0 and abs(minD)+maxD
    raw=raw./(maxD-minD); % the values are between 0 and 1 
    tweet_word_N=size(raw,1); % number of words in tweet
    if tweet_word_N>maxLength  % long tweets are cut 
        raw=raw(1:maxLength,:); % the first maxLength words are keep
    end
    tweet_word_N=size(raw,1);
    if tweet_word_N<maxLength % short tweets are filled with 0's
        eksiksayi=maxLength-tweet_word_N;
        raw=[raw ; zeros(eksiksayi,embeddingDimension)]; 
    end
    TRAIN_TW(:,:,1,i)=raw;
end

for i=1:test_N
    raw = word2vec(myemb,string(test_docs(i)));
    var_i=find(isnan(raw(:,1))==0);
    raw=raw(var_i,:); % words having wordembedding
    raw=raw+abs(minD); % add min. the values are between 0 and abs(minD)+maxD
    raw=raw./(maxD-minD); % the values are between 0 and 1 
    tweet_word_N=size(raw,1);
    if tweet_word_N>maxLength  % long tweets are cut 
        raw=raw(1:maxLength,:); % the first maxLength words are keep
    end
    tweet_word_N=size(raw,1);
    if tweet_word_N<maxLength % short tweets are filled with 0's
        eksiksayi=maxLength-tweet_word_N;
        raw=[raw ; zeros(eksiksayi,embeddingDimension)]; 
    end
    TEST_TW(:,:,1,i)=raw;
end
toc
layers = [imageInputLayer([image_row_size image_col_size 1]);
          convolution2dLayer([cnn_filter_width embeddingDimension],cnn_filter_N,'padding',[1 1],'stride',1);
          reluLayer();
          maxPooling2dLayer([maxLength-1 1],'stride',1);
          dropoutLayer('Probability',drop_prob);
          fullyConnectedLayer(numClasses); % number of classes
          softmaxLayer();
          classificationLayer()];
      
options = trainingOptions('sgdm','MaxEpochs',maxEpochs,...
	'InitialLearnRate',0.01);

convnet = trainNetwork(TRAIN_TW,train_labels,layers,options);

tic
YTest = classify(convnet,TEST_TW);
accuracy = sum(YTest == test_labels)/numel(test_labels)
toc
      
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




