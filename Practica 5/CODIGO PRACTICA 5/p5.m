clear ; 
close all;
load ('ex5data1.mat');
m = size(X, 1);
%utilizamos who para ver que variables se han incorporado al entorno
printf('variables se han incorporado al entorno.\n');
%llamada a la función who
who

% Plot
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

printf('Presione Entrer para continuar.\n');
pause;

lambda = 1;
theta = [1 ; 1];

%Funcion de coste y gradiente regularizado
[J, grad] = CosteGradRegu([ones(m, 1) X], y, theta, lambda);

printf('Coste con theta = [1 ; 1]: %f ', J);
printf('Gradiente con theta = [1 ; 1]:  [%f; %f] ',grad(1), grad(2));

printf('\n Presione Entrer para continuar.\n');
pause;

% Entrenamiento regresion lineal con lambda = 0
lambda = 0;
[theta] = entrenamientoLinReg([ones(m, 1) X], y, lambda);

%  Plot 
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '-', 'LineWidth', 2)
hold off;

printf('\n Presione Entrer para continuar.\n');
pause;

%2. Curvas de aprendizaje
lambda = 0;
%lambda = 1;
%lambda = 100;
[error,errorVali]=curvaAprendizaje([ones(m,1) X],y,[ones(size(Xval,1),1) Xval],yval,lambda);

plot(1:m, error, 1:m, errorVali);
title('Curva de aprendizaje para la regresión lineal')
legend('Entrenamiento', 'Validacion')
xlabel('Numero de ejemplos de entrenamiento')
ylabel('Error')

printf('\n Presione Entrer para continuar.\n');
pause;

%3. Regresión polinomial
p = 8;
% Normalizar
Xpoli = datosPoli(X, p);
[Xpoli, mu, sigma] = featureNormalize(Xpoli);  % Normalize

% Añadir unos
Xpoli = [ones(m, 1), Xpoli]; 
                 
%entrenamiento
lambda = 0;
[theta] = entrenamientoLinReg(Xpoli, y, lambda);
figure(1);
%pot
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Cambio en el nivel del agua (x)');
ylabel('Agua que derrama la presa (y)');
title (sprintf('Regresión Polinómica (lambda = 0)'));

printf('\n Presione Entrer para continuar.\n');
pause;
%para calcular el error sobre los ejemplos de validación Xval debes aplicarles  
%la misma transformación que a los de entrenamiento,generando las pot. desde 1 
%hasta p y normalizándolas luego usando las medias y desviaciones estándar 
%calculadas para los ejemplos de entrenamiento 
X_poly_val = datosPoli(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
 % Añadimos unos
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];   

%A continuación, puedes probar el efecto que tiene el término de regularización 
%repitiendo el proceso para ? = 1 y ? = 100.
% plots
lambda = 0;
%lambda = 1;
%lambda = 100;
[error, errorVali] = curvaAprendizaje(Xpoli, y, X_poly_val, yval, lambda);
figure(2);
plot(1:m, error, 1:m, errorVali);
title(sprintf('Curva de aprendizaje para la regresión polinomial(lambda = 0)'));
xlabel('Numero de ejemplos de entrenamiento')
ylabel('Error')
legend('Entrenamiento', 'Validación')

printf('\n Presione Entrer para continuar.\n');
pause;

%4. Selección del parámetro lambda
lambdaV = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
[error, errorVali] =validacion(Xpoli, y, X_poly_val, yval,lambdaV);

figure(3);
plot(lambdaV, error, lambdaV, errorVali);
legend('Entrenamiento', 'Validación');
xlabel('lambda');
ylabel('Error');
printf('\n Presione Entrer para continuar.\n');
pause;

printf('Error de validacion para lambda=3: %f\n', errorVali(9));

