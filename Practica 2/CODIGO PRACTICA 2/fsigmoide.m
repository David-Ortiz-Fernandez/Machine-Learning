%1.2. Funci�n sigmoide
function g = fsigmoide(z)
%la funci�n se puede aplicar a une escalar, matriz o vector
g = 1./ (1 + e.^-z); 

endfunction