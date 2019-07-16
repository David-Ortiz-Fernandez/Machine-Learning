clear all;
close all;
%Visualización de los datos
%cargamos los datos
data = load('ex2data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%Pintamos los datos
figure; 
hold on;
positivos =find (y==1);
negativos = find(y==0);

plot(X(positivos, 1), X(positivos, 2), 'k+','LineWidth', 3, 'MarkerSize', 7)
plot(X(negativos,1), X(negativos, 2),'ko','MarkerFaceColor','y')
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;

%2.1. Mapeo de los atributos
X = mapFeature(X(:,1), X(:,2));

%2.2. Cálculo de la función de coste y su gradiente
thetainicial = zeros(size(X, 2), 1);
lambda = 1;

[cost, grad] = costeReg(thetainicial, X, y, lambda);
printf('Coste para valores theta iniciales: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;

%2.3. y 2.4. Cálculo del valor óptimo de los parámetros con fminunc
%lambda = 0;
%lambda = 1;
lambda = 2;
%lambda = 3;
%lambda = 10;
%lambda = 20
%lambda = 50;
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J] = fminunc(@(t)(costeReg(t, X, y, lambda)), thetainicial, opciones);

printf('Coste para valores theta calculados con fminunc: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;


plotDecisionBoundary(theta, X, y);
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

p = porcentaje(theta, X, y);
