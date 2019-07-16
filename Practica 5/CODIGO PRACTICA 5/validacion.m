%funcion que genera los valores de error de entrenamiento y validacion para 
%un conjunto lambda que entra como parametro
function [error, errorVali] = validacion(X, y, Xval, yval,lambdaV)

% inicialización de variables a retornar
error= zeros(length(lambdaV), 1);
errorVali = zeros(length(lambdaV), 1);

%generacion de los valores de error para todos los lambda dados
for i = 1:length(lambdaV)
    lambda = lambdaV(i);
    theta = entrenamientoLinReg(X,y,lambda);
    error(i) = CosteGradRegu(X,y,theta,0);
    errorVali(i)   = CosteGradRegu(Xval,yval,theta,0);
endfor

endfunction
