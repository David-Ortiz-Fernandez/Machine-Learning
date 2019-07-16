function [J, grad] = coste(theta, X, y)  
m = length(y); %numero de ejemplos de entrenamiento

J = 0;
grad = zeros(size(theta));

h = fsigmoide(X*theta); %calculo de h con la función sigmoide

% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
J = (1/m)*(-y'* log(h) - (1 - y)'* log(1-h));
grad = (1/m)*X'*(h - y);

endfunction