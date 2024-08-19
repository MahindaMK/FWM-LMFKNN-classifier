close all; clear all; clc

% Load the data
original_data = readtable('appendicitis.dat'); 


% Convert table data to an array
data = table2array(original_data);
data(data(:,end)==0,end)=2; % replace class 0 with 2

% normalize the data into unit interval
X = data(:,1:end-1); % features
labels = data(:,end) ; % class variable

X = normalize(X,'range');

data = [X labels];

% Cross validation
val = 0.8; % Percentage for holdout validation
cv  = cvpartition(size(data,1),'HoldOut', val);
idx = cv.test;

% Separate to training and test data
Xtrain  = data(~idx,1:end-1); % train data with n patterns and m features
Ytrain  = data(~idx,end); % class labels of train patters 

Xtest   = data(idx,1:end-1); % test data with D patterns and m features
Ytest   = data(idx,end); % class labels of test patterns

% Parameter initialization
K = 4; % Initialization of the number of nearest neighbors (this can be tested for different values)
e = 2; % Minkoski distance parameter (this can be tested for different values)


% Function call for (relevance + complementarity)-based feature weights
PP = 1;
[~, index_rem, Hrlc] = feat_sel_FES_RRCom([Xtrain Ytrain], 'luca', PP);
        
feature_weights      = Hrlc;
        
% Function call for FWM-LMFKNN classifier     
[Predicted, membershios, accuracy] = FWM_LMFKNN(Xtrain, Ytrain, Xtest, Ytest, K, feature_weights, e);

classification_accuracy = accuracy
