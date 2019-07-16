%a función que devuelvael coste y el gradiente de la regresión lineal 
%regularizada a partir de los valores de entrada X, salida y, el vector de 
%parámetros ? y el valor de ?
function [J, grad] = CosteGradRegu(X, y, theta, lambda)

J = 0;
grad = zeros(size(theta));
m = length(y);
%calcul ode hipotesis
h = X*theta;
dif = h - y;

thetaReg = [0 ; theta(2:end, :)];

%calculo de función de coste
p = lambda*(thetaReg'*thetaReg);

J = (dif'*dif)/(2*m) + p/(2*m);

% calculo del gradiente
grad = (X'*dif+lambda*thetaReg)/m;

grad = grad(:);

endfunction