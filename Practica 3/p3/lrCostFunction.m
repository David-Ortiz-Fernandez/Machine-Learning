%Función que devuelva el coste y el gradiente de la regresión logística 
%regularizada sin utilizar bucles
function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));

% calculo de la hipotesis usando funcion sigmoide
h = fsigmoide(X*theta);

% regularizacion de theta eliminando primer valor
%thetaReg = [0;theta(2:end, :);];
theta(1) = 0;
% Vectorización de la función de coste
%J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*thetaReg'*thetaReg;
J = (1/m)*sum(-y.*log(h) .-(1.-y).*log(1.-h)) + ((lambda/(2*m))*sum(theta.^2));

%Vectorización del gradiente
grad = (1/m)*(X'*(h-y)+lambda*theta);

endfunction
