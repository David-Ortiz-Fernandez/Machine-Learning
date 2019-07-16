function [error, errorVali] = curvaAprendizaje(X, y, Xval, yval, lambda)
m = size(X, 1);
% error del resultado aplicado sobre esemismo subconjunto, y el error al 
% clasificar a los ejemplos del conjunto de validación
error = zeros(m, 1);
errorVali   = zeros(m, 1);

for i= 1:m
    theta = entrenamientoLinReg(X(1:i,:), y(1:i), lambda);
    %introducimos el valor lambda=0 par suprimir l regularizacion
    error(i) = CosteGradRegu(X(1:i,:), y(1:i), theta, 0);
    errorVali(i)   = CosteGradRegu(Xval, yval, theta, 0);
endfor

endfunction