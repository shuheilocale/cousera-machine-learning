function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

M = (1/m);


% for 1.2.3 Learning parameters using fminunc
%J = (1/m) * ( sum( -y .* log( sigmoid(X*theta) ) - ( 1 .- y ) .* log( 1 - sigmoid(X*theta ) ) ) );
%for i=1:m
%grad(i) = (1/m) * ( sum( sigmoid(X*theta) .- y ) * X(i) );
%endfor

% for 1.2.2 Cost function and gradient
sig = sigmoid(X * theta);

% = J 
J2 = sum((1/m) * (  -y .* log( sig ) - ( 1 - y ) .* log( 1 - sig ) ));

grad = (1/m) * (  (sig - y)' * X )';


J = 1 / m * (-y' * log(sig) - (1 - y') * log(1 - sig));

%grad = 1 / m * ((sig - y)' * X)';



% =============================================================

end
