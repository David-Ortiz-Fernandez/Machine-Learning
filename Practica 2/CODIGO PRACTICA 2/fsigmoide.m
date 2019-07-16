%1.2. Función sigmoide
function g = fsigmoide(z)
%la función se puede aplicar a une escalar, matriz o vector
g = 1./ (1 + e.^-z); 

endfunction