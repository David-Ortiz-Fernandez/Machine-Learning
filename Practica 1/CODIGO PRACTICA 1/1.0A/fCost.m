%Funcion que calcula el coste para unos parametros theta dados
function J = fCost(X, y, theta)
%matriz m*2 por matriz 2*1, asi calculamos los m valores de la hipotesis para 
%los theta0 y theta1 dados 
h = X*theta;
            
fCuadrado  = (h - y).^2;

m = length(y);
J = 1/(2*m) * sum(fCuadrado);

endfunction