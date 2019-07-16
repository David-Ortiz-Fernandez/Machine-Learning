clear ; 
close all;
%1.1. Visualización de los datos
load('ex3data1.mat');
% almacena los datos leídos en X, y
%m son 5000 elementos
m = size(X, 1);

% Selecciona aleatoriamente 100 ejemplos
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

printf('Presione enter para continuar.\n');
pause;
%1.2. Vectorización de la regresión logística
%Entrenamiento
num_etiquetas = 10;  
lambda = 0.1;

[all_theta] = oneVsAll(X, y, num_etiquetas, lambda);

printf('Presione enter para continuar.\n');
pause;
 
%1.3. Clasificación de uno frente a todos 
%Parte prediccion 
m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

%calcular para cada ejemplo de entrenamiento cuál es la probabilidad de que 
%pertenezca a cada una de las clases
predict = fsigmoide(X*all_theta');

% asignamos la etiqueta para la que se obtenga el valor máximo
[predict_max, index_max] = max(predict, [], 2);
prob = index_max;

printf('Predicción: %f\n', mean(double(prob == y)) * 100);