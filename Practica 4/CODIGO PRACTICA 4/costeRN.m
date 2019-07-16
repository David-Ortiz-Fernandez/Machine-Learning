function [J grad] = costeRN(params_rn, num_entradas, num_ocultas,num_etiquetas, 
                              X, y, lambda)
%reconstruir las matrices de parámetros a partir del vector params_rn
Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
                 num_ocultas, (num_entradas + 1));

Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), ...
                 num_etiquetas, (num_ocultas + 1));

% variables utililes
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
m = size(X, 1);

% para poder entrenar la red neuronal es necesario codificar las etiquetas como 
%vectores de 10 componentes con todos sus elementos a 0 salvo uno a 1
I = eye(num_etiquetas);
Y = zeros(m, num_etiquetas);
for i=1:m
  Y(i, :)= I(y(i), :);
end

% feedforward
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) fsigmoide(z2)];
z3 = a2*Theta2';
a3 = fsigmoide(z3);
h = a3;

% calculo de la penalizacion
p = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));

% calculo de J, si lambda es 0 se devolvera sin el termino regularizado
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*p/(2*m);

% calculo de sigmas
sigma3 = a3.-Y;
sigma2 = (sigma3*Theta2).*fsigmoideGradiente([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);

%calculo de deltas acumuladas
delta_2 = (sigma3'*a2);
delta_1 = (sigma2'*a1);

%si lambda es 0 calculo se devolvera el gradiente sin regularizar ya que depende
%de lambda
p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;

% el gradiente se devuelve desplegado como un vector columna
grad = [Theta1_grad(:) ; Theta2_grad(:)];
endfunction

