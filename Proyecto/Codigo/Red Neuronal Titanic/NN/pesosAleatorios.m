%a una función que inicialice una matriz de pesos T(l) con valores aleatorios en
%el rango [-epsilon_ini, epsilon_ini]. 
function W = pesosAleatorios(L_in, L_out)

W = zeros(L_out, 1 + L_in);
epsilon_ini = 0.12;
%epsilon_ini = 0.67;

%W debe tener tamaño (L_out, 1 + L_in)
W = rand(L_out, 1 + L_in) * 2 * epsilon_ini - epsilon_ini; 
%W = rand(L_out, L_in) * (2 * 0.12) - 0.12;
endfunction