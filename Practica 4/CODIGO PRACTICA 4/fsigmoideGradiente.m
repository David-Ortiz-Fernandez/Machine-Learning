%función que calcula la derivada del sigmoide tanto para un valor concreto,
%como para un vector o una matriz en cuyo caso lo aplicará a cada uno de sus 
%componentes.
function F = fsigmoideGradiente(z)

F = zeros(size(z));

F  = fsigmoide(z).*(1-fsigmoide(z));

endfunction