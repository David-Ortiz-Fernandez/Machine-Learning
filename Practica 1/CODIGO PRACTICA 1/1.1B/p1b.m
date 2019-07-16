clear;
%Cargamos los datos
data = load("ex1data2.txt");
X = data(:, 1:2); %features
y = data(:, 3); %target
m = length(y);

%valores sigma e iteraciones
alpha=zeros(4,1);
iters = 1500;

alpha(1) = 0.01; 
alpha(2) = 0.1; 
alpha(3) = 0.03;
alpha(4) = 0.3;

%normalizamos las features
[X,mu,sigma] = normalizaAtributo(X);
X = [ones(m, 1) X];

%descenso de gradiente
theta = zeros(3, 1);
[JS4, thetaJ4] = dGradienteMul(X, y, theta, alpha(4), iters);
[JS3, thetaJ3] = dGradienteMul(X, y, theta, alpha(3), iters);
[JS2, thetaJ2] = dGradienteMul(X, y, theta, alpha(2), iters);
[JS1, thetaJ1] = dGradienteMul(X, y, theta, alpha(1), iters);

% Plot
figure;
xlabel('Numero de iteraciones');
ylabel('Coste J');
plot(1:numel(JS4), JS4, '-r', 'LineWidth', 2);
hold on;
plot(1:numel(JS3), JS3, '-g', 'LineWidth', 2);
hold on;
plot(1:numel(JS2), JS2, '-r', 'LineWidth', 2);
hold on;
plot(1:numel(JS1), JS1, '-b', 'LineWidth', 2);
hold on;

%Valores de ThetaJ
printf("Valores Theta0: %f ,Theta1: %f ,Theta2: %f \n", thetaJ4(1), thetaJ4(2),thetaJ4(3));
printf("Pulse Enter para continuar\n")
pause;

%calculo para 1.650 pies cuadrados y 3 habitaciones
precio = [1 (1650-mu(1))/sigma(1) (3-mu(2))/sigma(2)]*thetaJ1;
printf("El precio estimado para una casa de 1650 pies cuadrados y tres habitaciones  es: %f \n",precio)

%

printf("Pulse Enter para finalizar.\n");
pause;

