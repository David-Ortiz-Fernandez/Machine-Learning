function J = fCost(X, y, theta)

m = length(y);

J = (1/(2*m)) * (X * theta - y)' * (X * theta - y);

endfunction
