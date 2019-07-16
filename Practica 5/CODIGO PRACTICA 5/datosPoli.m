%Esta funci�n recibir� una matriz X de dimensi�n m � 1 y un n�mero p, y 
%devolver� otra matriz de dimensi�n m � p que en la primera columna contenga 
%los valores de X, en la segunda el resultado de calcular X.^2, en la tercera
%X.^3, y as� sucesivamente.
function Xpoli = datosPoli(X, p)

Xpoli = zeros(numel(X), p);
Xpoli(:,1) = X;

for i=2:p
    Xpoli(:,i) = X.*Xpoli(:,i-1);
endfor


endfunction