function [JS, theta] = dGradienteMul(X, y, theta, alpha, num_iters)
m = length(y); 

  for iter = 1:num_iters
      theta = theta - (alpha/m) * (X' * (X * theta - y));
      JS(iter) = fCost(X, y, theta);
  endfor

endfunction
