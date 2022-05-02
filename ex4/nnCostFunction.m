function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m,1) X]; % 5000x401 (add a column of ones)
a2 = sigmoid(Theta1*X'); % 25x401 * 401x5000 = 25x5000

a2 = [ones(m,1) a2']; % 5000x26 (add a column of ones)
a3 = sigmoid(Theta2*a2'); % 10x26 * 26x5000 = 10x5000

h_theta = a3; % 10x5000

% Vectorizing output y
y_new = zeros(num_labels,m); % 10x5000
for i=1:m
    y_new(y(i),i)=1;
end

% Cost function J
J = (-1/m) * sum(sum(y_new .* log(h_theta) + (1-y_new) .* log(1-h_theta)));



%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

for t=1:m
    
   % Step 1
   a1 = X(t,:)'; % a1 = t-th row of X / transposed (401x1)
   
   z2 = Theta1 * a1; % (25x1)
   a2 = [1 ; sigmoid(z2)]; % (26x1)
   
   z3 = Theta2 * a2; % (10x1)
   a3 = sigmoid(z3); % (10x1)
   
   % Step 2
   delta_3 = a3 - y_new(:,t); % (10x1)
   
   % Step 3
   z2 = [1 ; z2]; % (26x1)
   delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % (26x1)
   
   % Step 4
   delta_2 = delta_2(2:end); % (25x1)
   
   Delta_1 = Delta_1 + delta_2 * a1';  % (25x401)
   Delta_2 = Delta_2 + delta_3 * a2';  % (10x26)
       
end

Theta1_grad = (1/m) * Delta_1;
Theta2_grad = (1/m) * Delta_2;



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Regularize the cost
Theta1(:,1) = []; % 25x400
Theta2(:,1) = []; % 10x25

A = Theta1.^2; % squares element-wise
B = Theta2.^2;

J = J + (lambda/(2*m)) * (sum(A,"all") + sum(B,"all"));

% Regularize the gradient
A = Theta1;
B = Theta2;
A = [zeros(size(A,1),1) A]; % add a first column of zeros
B = [zeros(size(B,1),1) B];


Theta1_grad = Theta1_grad + (lambda/m) * A;
Theta2_grad = Theta2_grad + (lambda/m) * B;



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
