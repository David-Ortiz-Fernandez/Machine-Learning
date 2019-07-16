clear; 
close all;
load('ex4data1.mat');
m = size(X, 1);
load('ex4weights.mat');
% Theta1 es de dimensión 25 x 401
% Theta2 es de dimensión 10 x 26

%desenrrollado de parametros
params_rn = [Theta1(:) ; Theta2(:)];

num_entradas  = 400;  % 20x20 
num_ocultas = 25;   
num_etiquetas = 10;

%al hacer el termino regularizado dependiente de lambda en la función costeRN
%introduciendo lambda=0 calcularemos el coste sin el termino regularizado 
lambda = 0;
J = costeRN(params_rn, num_entradas, num_ocultas,num_etiquetas, X, y, lambda);

printf('Coste sin regularizar: %f\n', J);

%al hacer el termino regularizado dependiente de lambda en la función costeRN
%introduciendo lambda=1 calcularemos el coste con el termino regularizado 
lambda = 1;
J = costeRN(params_rn, num_entradas, num_ocultas,num_etiquetas, X, y, lambda);
printf('Coste regularizado: %f\n' , J);
printf('Presione enter para continuar.\n');
pause;


fprintf('Inicializando parametros\n')
Theta1_inicial = pesosAleatorios(num_entradas, num_ocultas );
Theta2_inicial = pesosAleatorios(num_ocultas , num_etiquetas);
% desenrrollado de parametros
params_rn_iniciales = [Theta1_inicial(:) ; Theta2_inicial(:)];
printf('Presione enter para continuar.\n');
pause;

%al hacer el termino regularizado dependiente de lambda en la función costeRN
%introduciendo lambda=0 como parametro de la función checkNNGradients haremos el
%chequeo del gradiente sin el termino regularizado
printf("Chequeo del gradiente sin regularizar. \n")
checkNNGradients(0);
printf('Presione enter para continuar.\n');
pause;

%al hacer el termino regularizado dependiente de lambda en la función costeRN
%introduciendo lambda>0 como parametro de la función checkNNGradients haremos el
%chequeo del gradiente con el termino regularizado
printf("Chequeo del gradiente regularizado. \n")
checkNNGradients(1);
printf('Presione enter para continuar.\n');
pause;

%Aprendizaje de los parámetros
printf('Entrenando la red con 50 iteraciones y un valor de lambda=1.\n');


%jugar con los valores  de lambda e iteraciones  para ver diferentes resultados
lambda = 1;
opciones = optimset('MaxIter', 50);

funcionCoste = @(p) costeRN(p,num_entradas,num_ocultas ,num_etiquetas,X,y,lambda)
[params_rn, cost] = fmincg(funcionCoste, params_rn_iniciales, opciones);

% obtenemos Theta1 y Theta 2 como en la practica anterior
Theta1 = reshape(params_rn(1:num_ocultas * (num_entradas + 1)), ...
                 num_ocultas, (num_entradas + 1));

Theta2 = reshape(params_rn((1 + (num_ocultas * (num_entradas + 1))):end), ...
                 num_etiquetas, (num_ocultas + 1));

printf('Presione enter para continuar.\n');
pause;

%calculo de la precisión de la red una vez entrenada
%forwardpropagation
p = zeros(size(X, 1), 1);

h1 = fsigmoide([ones(m, 1) X] * Theta1');
h2 = fsigmoide([ones(m, 1) h1] * Theta2');

%calculo de la precisión de la red una vez entrenada
[pmaxi, p] = max(h2, [], 2);
printf('Precisión: %f\n', mean(double(p == y)) * 100);
