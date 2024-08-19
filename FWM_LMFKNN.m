function [y_predicted, memberships, accuracy] = FWM_LMFKNN(Xtrain, Ytrain, Xtest,Ytest, k_values, w, e)

%--------------------------------------------------------------------------
% LM_FKNN method

% INPUT:
% Xtrain  = train data
% Xtest   = test data
% Ytrain  = train classes
% Ytest   = test classes
% k_values       = number of nearest neighbors
% w       = feature weights
% e       = Minkowski distance parameter

% OUTPUT:
% y_predicted = predicted class labels
% memberships = class membership results
% accuracy    = classification accuracy over test data set
%--------------------------------------------------------------------------

% initialization
num_test  = size(Xtest,1);
max_class = max(Ytrain);
m = 2;

% for each test point, do:

for i=1:num_test
    clas_index = unique(Ytrain);
        
    for ii=1:length(clas_index)
    train_data_class_ii = Xtrain(Ytrain==clas_index(ii),:);
    num_train_ii        = size(train_data_class_ii,1);
      

    for b=1:num_train_ii
       distances(b) = (sum(w.*abs(Xtest(i,:)-train_data_class_ii(b,:)).^e).^(1/e));
    end

    % sort the similarity values 
     [~, indeces] = sort(distances);


    if (num_train_ii<k_values)
    neighbor_index = indeces;    
    else
    neighbor_index = indeces(1:k_values);
    end
	weight = ones(1,length(neighbor_index));
        
    lm_vector = mean(train_data_class_ii(neighbor_index,:),1);
    localmean(ii,:) = lm_vector;
    labels3(ii)     =  clas_index(ii);
    clear distances
    end
    
    data2  = localmean;
    [n1,~] = size(data2);
    
    
    for b=1:n1
       distances(b) = (sum(w.*abs(Xtest(i,:)-data2(b,:)).^e).^(1/e));
    end

 
    weight = distances.^(-1/(m-1));
 
 	% set the Inf (infite) weights, if there are any, to  1.
 	if max(isinf(weight))
        weight(isinf(weight))=1;
 	end

    
    % convert class labels to unary membership vectors (of 1s and 0s)
    labels_iter = zeros(length(labels3),max_class);
    for ii=1:n1
        labels_iter(ii,:) = [zeros(1, labels3(ii)-1) 1 zeros(1,max_class - labels3(ii))];
    end    
    
	test_out = weight*labels_iter/(sum(weight));
    
    memberships(i,:) = test_out; 
    [~, class_idx]   = max(test_out');
    y_predicted(i)   = class_idx; % predicted class
    
    clear labels3;  clear labels_iter;

    
end

accuracy = sum(y_predicted'==Ytest)/length(Ytest);
     

