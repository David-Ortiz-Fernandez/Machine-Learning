clear;
%% Load Data
data = load("ex1data2.txt");
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

X = [ones(m, 1) X];

% calculo de ThetaJ con la ecuación normal
thetaJ = normalEqn(X, y);

%Valores de ThetaJ
printf("Valores Theta0: %f ,Theta1: %f ,Theta2: %f \n", thetaJ(1), thetaJ(2),thetaJ(3));
printf("Pulse Enter para continuar\n")
pause;

%calculo para 1.650 pies cuadrados y 3 habitaciones
precio = [1 1650 3]*thetaJ; 
printf("El precio estimado para una casa de 1650 pies cuadrados y tres habitaciones  es: %f \n",precio)

pause;

