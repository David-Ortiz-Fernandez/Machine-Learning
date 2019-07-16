%función que devuelva una matriz T ? RK×(N+1) donde cada fila de T corresponde 
%a los parámetros aprendidos para el clasificador de una de las clases
function [all_theta] = oneVsAll(X, y, num_etiquetas, lambda)
%ONEVSALL entrenavarios clasificadores por regresión logística y devuelve
% el resultado en una matriz all_ t h e t a , donde la fila i-ésima
% corresponde al clasificador de la etiqueta i-ésima

m1 = size(X, 1);
m2 = size(X, 2);

all_theta = zeros(num_etiquetas, m2 + 1);

X = [ones(m1, 1) X];

initial_theta = zeros(m2 + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

  for c = 1:num_etiquetas,
    
   [theta] = fmincg(@(t)(lrCostFunction(t, X, (y==c), lambda)), initial_theta, options);
    all_theta(c, :) = theta';
    
  endfor

endfunction
