close all;
clear ; 

load('ex3data1.mat');
m = size(X, 1);
% El ficherochero ex3weights.mat contiene las matrices T(1) y T(2) con el 
%resultado de haber entrenado la red neuronal
load('ex3weights.mat');
% Theta1 es de dimensión 25 x 401
% Theta2 es de dimensión 10 x 26

%computar el valor de h?(x(i)) paracada ejemplo i. De la misma forma que en la 
%regresión logística, interpretaremos que la clase asignada por la red neuronal 
%a un ejemplo es la correspondiente a la salida de la red con el máximo valor 
%propagación hacia delante
X = [ones(size(X,1),1) X];

h = fsigmoide(X * Theta1');
h = [ones(size(h,1),1) h];

ma = fsigmoide(h * Theta2');
[x, pre] = max(ma');

pre = pre';
printf('La precisión de la red neuronal es: %f\n', mean(double(pre == y))*100);


