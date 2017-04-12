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

% for 2.3 Cost function and gradient

sig = sigmoid(X*theta);

reg = lambda / (2 * m) * (theta' * theta - theta(1)^2);

% lambda / 2*m * sum(theta .^2);

J = (1/m) * ( sum( -y .* log( sig ) - ( 1 .- y ) .* log( 1 - sig ) ) ) + reg;

%disp(J2);


%grad(1) = (1/m) * ( sum( sig .- y ) * X(1) );

%for i=2:length(theta)
%  grad(i) = (1/m) * ( sum( sig .- y ) * X(i) ) + lambda / m * theta(i);
%endfor


%%%

%sig = sigmoid(X * theta); 

%reg = lambda / (2 * m) * (theta' * theta - theta(1)^2);

%J = 1 / m * (-y' * log(sig) - (1 - y') * log(1 - sig)) + reg;

mask = ones(size(theta));
mask(1) = 0;

grad = 1 / m * X' * (sig - y) + lambda / m * (theta .* mask);




% =============================================================

end
