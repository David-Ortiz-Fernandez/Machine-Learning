%Esta función recibirá una matriz X de dimensión m × 1 y un número p, y 
%devolverá otra matriz de dimensión m × p que en la primera columna contenga 
%los valores de X, en la segunda el resultado de calcular X.^2, en la tercera
%X.^3, y así sucesivamente.
function Xpoli = datosPoli(X, p)

Xpoli = zeros(numel(X), p);
Xpoli(:,1) = X;

for i=2:p
    Xpoli(:,i) = X.*Xpoli(:,i-1);
endfor


endfunction