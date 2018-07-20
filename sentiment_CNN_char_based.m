% (c) Mehmet Fatih Amasyali, 2018
clear all;
close all;
rng(1);
% sentiment analysis (actually text classification, you can change train and test files for your text classification application)

% Method: convert tweets (char based) to images then train with CNNs 

% Train and test files are in excel format. 
% first column: tweet text 
% second column: tweet label (positive, negative, notr)
trainfile='train_tweets.xlsx'; 
testfile='test_tweets.xlsx';
min_ch_fre=20; % min_frequence of chars 
maxLength=150; % max char count in a tweet, for longer tweets the first maxLength chars are keep 

cnn_filter_width=3; % width of cnn filters
cnn_filter_N=200; % number of cnn filters
maxEpochs=200; % epoch for CNN training
drop_prob=0.66; % dropout prob.
labels=[ "olumsuz", "notr", "olumlu"];
numClasses = size(labels,2); 

train_table=readtable(trainfile,'ReadVariableNames',false);
test_table=readtable(testfile,'ReadVariableNames',false);
train_tweets=string(table2cell(train_table(:,1)));
test_tweets=string(table2cell(test_table(:,1)));
train_labels=categorical(table2cell(train_table(:,2)));
test_labels=categorical(table2cell(test_table(:,2)));

train_docs = lower(train_tweets); % convert tweets to lowercase
test_docs = lower(test_tweets); 

allC = unique(char(train_docs)); 
CNum=size(allC,1); % number of unique character 

hist_c=zeros(1,CNum);
for i=1:CNum
hist_c(i) = sum(count(train_docs,allC(i)));
end 
allC=allC(hist_c>min_ch_fre);
CNum=size(allC,1); % number of unique character after infrequent char elimination 

train_N=size(train_labels,1);
test_N=size(test_labels,1);

embeddingDimension = CNum;
image_row_size=maxLength;
image_col_size=embeddingDimension;
% encode each char with one hat vectors
% char embeddings is a Csayi*Csayi dimensional matrix 
Cemb=zeros(CNum,CNum);
for i=1:CNum
    Cemb(i,i)=1;
end
TRAIN_TW_c=zeros(image_row_size,image_col_size,1,train_N);
TEST_TW_c=zeros(image_row_size,image_col_size,1,test_N);
tic
for i=1:train_N
    [~,Locb] = ismember(double(char(train_docs(i))),double(allC')); % for each char in i.th doc, find indices in allC 
    Locb=Locb((Locb~=0)); % delete chars which does not exists in allC
    raw=cat(1,Cemb(Locb,:)); % raw is a number of chars in tweet * allC dimensional matrix 
    tweet_char_N=size(raw,1); % number of chars in tweet
    if tweet_char_N>maxLength  % long tweets are cut 
        raw=raw(1:maxLength,:); % the first maxLength chars are keep
    end
    tweet_char_N=size(raw,1);
    if tweet_char_N<maxLength % short tweets are filled with 0's
        eksiksayi=maxLength-tweet_char_N;
        raw=[raw ; zeros(eksiksayi,embeddingDimension)]; 
    end
    TRAIN_TW_c(:,:,1,i)=raw;
end

for i=1:test_N
    [~,Locb] = ismember(double(char(test_docs(i))),double(allC')); % for each char in i.th doc, find indices in allC 
    Locb=Locb((Locb~=0)); % delete chars which does not exists in allC
    raw=cat(1,Cemb(Locb,:)); % raw is a number of chars in tweet * allC dimensional matrix 
    tweet_char_N=size(raw,1); % number of chars in tweet
    if tweet_char_N>maxLength  % long tweets are cut
        raw=raw(1:maxLength,:); % the first maxLength chars are keep
    end
    tweet_char_N=size(raw,1);
    if tweet_char_N<maxLength % short tweets are filled with 0's
        eksiksayi=maxLength-tweet_char_N;
        raw=[raw ; zeros(eksiksayi,embeddingDimension)]; 
    end
    TEST_TW_c(:,:,1,i)=raw;
end

toc
layers = [imageInputLayer([image_row_size image_col_size 1]);
          convolution2dLayer([cnn_filter_width embeddingDimension],cnn_filter_N,'padding',[1 1],'stride',1);
          reluLayer();
          maxPooling2dLayer([maxLength-2 1],'stride',1);
          dropoutLayer('Probability',drop_prob);
          fullyConnectedLayer(numClasses); % number of classes
          softmaxLayer();
          classificationLayer()];
options = trainingOptions('sgdm','MaxEpochs',maxEpochs,...
	'InitialLearnRate',0.01);

convnet = trainNetwork(TRAIN_TW_c,categorical(train_labels),layers,options);

tic
YTest = classify(convnet,TEST_TW_c);
accuracy = sum(YTest == categorical(test_labels))/numel(categorical(test_labels))
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
