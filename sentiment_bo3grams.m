% (c) Mehmet Fatih Amasyali, 2018
clear all;
close all;
rng(1); 
% sentiment analysis (actually text classification, you can change train and test files for your text classification application)

% Method: Create term (char 3grams) document matrix from train and test files. 
% Then writes to a sparse arff file.  
% the full term document matrix can be very large. It can not be handled with Matlab. 
% for this reason, a sparse matrix is used. But, Matlab algorithms can be
% worked with sparse matrix are very limited.
% So the sparse matrix is written to a sparse arff file.
% All Weka algorithms can be used with sparse matrixex.   

% Train and test files are in excel format. 
% first column: tweet text 
% second column: tweet label (positive, negative, notr)
trainfile='train_tweets.xlsx'; 
testfile='test_tweets.xlsx';
min_fre=5; % min_frequence of char 3grams 
arfffile='sentiment_bo3GR_8020.arff';

train_table=readtable(trainfile,'ReadVariableNames',false);
test_table=readtable(testfile,'ReadVariableNames',false);
train_tweets=string(table2cell(train_table(:,1)));
test_tweets=string(table2cell(test_table(:,1)));
train_labels=string(table2cell(train_table(:,2)));
test_labels=string(table2cell(test_table(:,2)));
all_labels=[train_labels;test_labels];

train_docs = lower(train_tweets); % convert tweets to lowercase 
test_docs = lower(test_tweets); 

train_N=size(train_labels,1);
test_N=size(test_labels,1);

% convert all train tweets to one char array 
tic
charTR = char(strjoin(train_docs));
% obtain unique char 3grams 
% modify below code to use 4grams, Xgrams instead of 3grams 
gr3 = strings(3,size(charTR,2)-2); 
for i=1:size(charTR,2)-2
    gr3(i)=charTR(i:i+2);
end
toc

tic
ugr3=unique(gr3); % unique 3grams are obtained, approx 3*10^4 :)
ugr3(1)=[]; % remove first empty element 
data_arff = sparse(train_N+test_N,size(ugr3,1));
f_ugr3=zeros(1,size(ugr3,1)); 
for i=1:size(ugr3,1)
    k=strfind(train_docs,ugr3(i));
    [~,ncols1] = cellfun(@size,k);
    f_ugr3(i) = sum(ncols1); % f_ugr3 contains frequencies of i.th 3gram in train files 
    k=strfind(test_docs,ugr3(i));
    [~,ncols2] = cellfun(@size,k);
    gr3_data=[ncols1;ncols2]; % gr3_data contains frequencies of i.th 3gram in all files 
    data_arff(:,i)=gr3_data;   
end
toc
plot(f_ugr3); 

s_gr3_ind=find(f_ugr3<min_fre); % indices of infrequent 3grams in train files 
disp(['number of unique 3grams in train files= ' num2str(size(ugr3,1))]);
data_arff(:,s_gr3_ind)=[]; % eliminate infrequent 3grams form all files.
ugr3(s_gr3_ind)=[];
disp(['number of unique 3grams after infrequent 3gram elimination= ' num2str(size(ugr3,1))]);

% write sparse arff 
tic
nV=size(data_arff,2);
nD=size(data_arff,1);
M = data_arff; 
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
    fprintf( fid,'%d %s %s\n',nV+1, all_labels(i),"}");
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
