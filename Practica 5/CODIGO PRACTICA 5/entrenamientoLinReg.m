function [theta] = entrenamientoLinReg(X, y, lambda)
  
initial_theta = zeros(size(X, 2), 1); 

costFunction = @(t) CosteGradRegu(X, y, t, lambda);

options = optimset('MaxIter', 200, 'GradObj', 'on');

% minimización usando fmincg
theta = fmincg(costFunction, initial_theta, options);

endfunction