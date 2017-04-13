function [J grad] = nnCostFunction(nn_params, ...
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% 10クラスと言う意味
K = num_labels;

%　yが正解値になってるのでこれをベクトル1=[1,0,0,0,..]みたいな
% 形式に変更する
Y = eye(K)(y, : );

bias = ones(m, 1);
a1 = [bias X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);

bias = ones(size(a2,1), 1);
a2 = [bias a2];

z3 = a2 * Theta2';
hx = sigmoid(z3);

% コスト計算
cost = sum((-Y .* log(hx)) - ((1 - Y) .* log(1 - hx)), 2);


Theta1_no_0 = Theta1(:, 2:end);
Theta2_no_0 = Theta2(:, 2:end);

reg = 0;
for j=1:size(Theta1_no_0,1)
  for k=1:size(Theta1_no_0,2)
       reg += ( Theta1_no_0(j,k) * Theta1_no_0(j,k)' );
    endfor
endfor

for j=1:size(Theta2_no_0,1)
  for k=1:size(Theta2_no_0,2)
       reg += ( Theta2_no_0(j,k) * Theta2_no_0(j,k)' );
    endfor
endfor


reg = ( lambda / (2 * m) ) * reg;


J = ( 1 / m ) * sum(cost) + reg;

% backpropagation


% Delta(l) values
Delta1 = 0;
Delta2 = 0;


% サンプル分だけ回す
for t = 1:m 
	% 1. input layer's value
	a1 = [1; X(t, :)'];
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% 2. output layer
	d3 = a3 - Y(t, :)';
	
	% 3. hidden layer
	d2 = (Theta2_no_0' * d3) .* sigmoidGradient(z2);

	% 4. gradient
  Delta1 += (d2 * a1');
	Delta2 += (d3 * a2');
endfor

% 5. theta gradient 
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;



Theta1_grad(:, 2:end) += lambda/m * Theta1_no_0;
Theta2_grad(:, 2:end) += lambda/m * Theta2_no_0;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
