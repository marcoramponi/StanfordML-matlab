function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

R = ones(1,3); % Matrix of results (3rd col = error for given choice of C,sigma)

for Cvec = [0.01 0.03 0.1 0.3 1 3 10 30]
    for sigmavec = [0.01 0.03 0.1 0.3 1 3 10 30]
        
        model = svmTrain(X, y, Cvec, @(x1, x2)gaussianKernel(x1, x2, sigmavec));
        predictions = svmPredict(model,Xval);
        error = mean(double(predictions ~= yval));
        result = [Cvec sigmavec error]; % row vec 1x3
        R = [R; result];
        
    end
end

R(1,:) = []; % remove first row (of ones) from R
R = sortrows(R,3); % sort rows of R (ascending) based on elements of 3rd column

C = R(1,1);
sigma = R(1,2);



% =========================================================================

end
