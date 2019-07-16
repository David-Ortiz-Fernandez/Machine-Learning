clear all;
close all;
%1.1. Visualizaci�n de los datos
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

%1.3. C�lculo de la funci�n de coste y su gradiente
[m, n] = size(X);
X = [ones(m, 1) X]; %como en regresi�n lineal a�adimos 1

% Inicializamos los valores de theta a 0
thetainicial = zeros(n + 1, 1);

% Coste inicial y gradiente
[cost, grad] = coste(thetainicial, X, y);

printf('Coste para valores theta iniciales: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;


%1.4. C�lculo del valor �ptimo de los par�metros con fminunc
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
% Obtenci�n d el valor �ptimo de theta
[theta, cost] = fminunc(@(t)(coste(t, X, y)), thetainicial, opciones);

printf('Coste para valores theta calculados con fminunc: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;

%para obtener la gr�fica utilizamos la funci�n plotDecisionBoundary
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%1.5. Evaluaci�n de la regresi�n log�stica
p = porcentaje(theta, X, y);