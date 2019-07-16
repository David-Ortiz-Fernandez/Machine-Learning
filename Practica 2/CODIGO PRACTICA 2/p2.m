clear all;
close all;
%1.1. Visualización de los datos
%cargamos los datos
data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%Pintamos los datos
figure; 
hold on;
positivos =find (y==1);
negativos = find(y==0);

plot(X(positivos, 1), X(positivos, 2), 'k+','LineWidth', 3, 'MarkerSize', 7)
plot(X(negativos,1), X(negativos, 2),'ko','MarkerFaceColor','y')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%1.3. Cálculo de la función de coste y su gradiente
[m, n] = size(X);
X = [ones(m, 1) X]; %como en regresión lineal añadimos 1

% Inicializamos los valores de theta a 0
thetainicial = zeros(n + 1, 1);

% Coste inicial y gradiente
[cost, grad] = coste(thetainicial, X, y);

printf('Coste para valores theta iniciales: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;


%1.4. Cálculo del valor óptimo de los parámetros con fminunc
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
% Obtención d el valor óptimo de theta
[theta, cost] = fminunc(@(t)(coste(t, X, y)), thetainicial, opciones);

printf('Coste para valores theta calculados con fminunc: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;

%para obtener la gráfica utilizamos la función plotDecisionBoundary
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%1.5. Evaluación de la regresión logística
p = porcentaje(theta, X, y);