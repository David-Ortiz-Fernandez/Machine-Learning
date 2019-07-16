function p = porcentaje(theta, X, y)
  m = size(X, 1); % numero de ejemplos de entrenamiento
  p = zeros(m, 1);
  
  p = (fsigmoide(X*theta) >= 0.5);
  
  printf('Porcentaje de aciertos: %f\n', mean(double(p == y)) * 100);
endfunction