function h = forwardpropagation(X, theta, num_entradas, num_ocultas, num_etiquetas)

  Theta1 = reshape(theta(1:num_ocultas * (num_entradas + 1)), num_ocultas, (num_entradas + 1));
  Theta2 = reshape(theta(1 + (num_ocultas * (num_entradas + 1)):end), num_etiquetas, (num_ocultas + 1));

  m = size(X, 1);
  a1=[ones(m,1), X];
  z2 = a1 * Theta1';
  a2 = fsigmoide(z2);
  a2 = [ones(size(a2,1), 1), a2]; 
  
  z3 = a2 * Theta2';
  a3 = fsigmoide(z3);
  
  h=a3;
endfunction


