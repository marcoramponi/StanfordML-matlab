function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


v = X*theta;
hvec = sigmoid(v);
Jvec = -y.*log(hvec) - (1-y).*log(1-hvec);
theta(1) = 0;
J = (1/m)*sum(Jvec) + (lambda/(2*m))*sum(theta.^2);

scarto = (hvec - y)';
grad = (1/m)*scarto*X + (lambda/m)*theta;





% =============================================================

end
