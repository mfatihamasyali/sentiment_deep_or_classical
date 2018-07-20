% (c) Mehmet Fatih Amasyali, 2018
clear all;
close all;
rng(1); 
% sentiment analysis (actually text classification, you can change train and test files for your text classification application)

% Method: Create term(word) document matrix from train and test files. 
% Then writes to an arff file.  

% Train and test files are in excel format. 
% first column: tweet text 
% second column: tweet label (positive, negative, notr)
trainfile='train_tweets.xlsx'; 
testfile='test_tweets.xlsx';
min_fre=5; % min_frequence of words 
arfffile='sentiment_bow_8020.arff';

train_table=readtable(trainfile,'ReadVariableNames',false);
test_table=readtable(testfile,'ReadVariableNames',false);
train_tweets=string(table2cell(train_table(:,1)));
test_tweets=string(table2cell(test_table(:,1)));
train_labels=string(table2cell(train_table(:,2)));
test_labels=string(table2cell(test_table(:,2)));

train_docs = tokenizedDocument(train_tweets);
train_docs = lower(train_docs); % convert docs to lowercase 
test_docs = tokenizedDocument(test_tweets);
test_docs = lower(test_docs); % 
tic
train_bag = bagOfWords(train_docs);
orj_bag=addDocument(train_bag,test_docs);
disp(['number of unique words in the original bag=' int2str(orj_bag.NumWords)]);
% preprocess
orj_bag = removeInfrequentWords(orj_bag,min_fre); % min_fre ten az kez gecenleri ele
disp(['number of unique words after infrequent word elimination=' int2str(orj_bag.NumWords)]);

orj_labels=[train_labels;test_labels];
nV=orj_bag.NumWords;
nD=orj_bag.NumDocuments;
M = orj_bag.Counts; % term document matrix
M(M>1)=1; % switch to binary (1/0) encoding from tf endocing. TF encoding is used if this line is commented 

fid = fopen(arfffile,'w'); 
fprintf( fid,'%s\n',"@relation ss");
for i=0:nV
    fprintf( fid,'%s%d %s\n',"@attribute x",i,"real");
end
fprintf( fid,'%s%d %s\n',"@attribute x",nV+1,"{olumlu, olumsuz, notr}");
fprintf( fid,'%s\n',"@data");
for i=1:nD
    [xi,xj,val] = find(M(i,:));
    fprintf( fid,'%s',"{");
    for j=1:size(xj,2)
        fprintf( fid,' %d %d, ', xj(j), val(j) );
    end
    fprintf( fid,'%d %s %s\n',nV+1, orj_labels(i),"}");
end
fclose(fid);
toc
disp('arff file was written. it is ready to use with Weka. The first 80% of data should be used as training data.');
disp('To do this, set the percentage split values as 80, than click -more option-s and mark -preserve order for % split- ');
disp('To avoid stack and heap problems, start weka with the following command "java -Xmx4G -Xss2048m -jar weka.jar" ');

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



