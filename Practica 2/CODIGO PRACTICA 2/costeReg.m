function [J, grad] = costeReg(theta, X, y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));

h = fsigmoide(X*theta);

stheta = theta(2:size(theta));
thetaReg = [0;stheta];

% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*thetaReg'*thetaReg;

grad = (1/m)*(X'*(h-y)+lambda*thetaReg);

endfunction
