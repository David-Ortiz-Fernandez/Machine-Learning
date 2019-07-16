%a una función que inicialice una matriz de pesos T(l) con valores aleatorios en
%el rango [-epsilon_ini, epsilon_ini]. 
function W = pesosAleatorios(L_in, L_out)

W = zeros(L_out, 1 + L_in);
epsilon_ini = 0.12;

%W debe tener tamaño (L_out, 1 + L_in)
W = rand(L_out, 1 + L_in) * 2 * epsilon_ini - epsilon_ini; 

endfunction