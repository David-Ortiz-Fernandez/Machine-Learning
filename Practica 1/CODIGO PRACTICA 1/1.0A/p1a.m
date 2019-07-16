%parametros iniciales 
clear;
iter = 1500;
alpha = 0.01;

% valores de theta0 theta1 iniciales son 0
theta0 = 0;
theta1 = 0;
theta = zeros(2, 1); %vector de tamaño 2x1

data = load("ex1data1.txt");

x = data(:, 1); %primera columna de datos de entrenamiento
y = data(:, 2); %segunda columna de datos de entrenamiento

printf("Pintando datos de entrenamiento \n")

plot(x, y, "rx", "MarkerSize", 8,"linewidth", 2); 
ylabel("Profit in $10,000s"); 
xlabel("Population of City in 10,000s");

printf("Pulse Enter para continuar\n")
pause;

% añadir 1s como primera componente de cada ejemplo de entrenamiento
m = length(x);
X = [ones(m, 1), x]; % para multiplicar Theta^T*X (producto vectorial)

%llamada a la funcion de coste con valores iniciales
costeInicial = fCost(X, y, theta);

[JS, thetaJ] = dGradiente(X, y, theta, alpha, iter);

printf("El valor que minimiza la función de coste es:  %f\n",min(JS));
printf("Valores Theta0: %f y Theta1: %f \n", thetaJ(1), thetaJ(2));
hold on;
printf("Pulse Enter para continuar\n")
pause;

printf("Pintando recta de regresion lineal... \n");

plot(x, X*thetaJ, "b-", "linewidth", 3)

hold off %de esta manera no machacamos plots posteriores

predic = thetaJ(1)+ 15000*thetaJ(2);
printf("La prediccion para una poblacion de 15000 habitantes es:  %f\n",predic);

predic = thetaJ(1)+ 75000*thetaJ(2);
printf("La prediccion para una poblacion de 75000 habitantes es:  %f\n",predic);

printf("Pulse Enter para continuar.\n");
pause;


vTheta0 = linspace(-10, 10, 100);
vTheta1 = linspace(-1, 4, 100);

for i = 1:length(vTheta0)
    for j = 1:length(vTheta1)
	  t = [vTheta0(i); vTheta1(j)]; 
    %Crear matriz de  funciones de costes 
	  Jniv(i,j) = fCost(X, y, t);
    end
end

%Hacemos la traspuesta para que la visualizacion sea correcta en los ejes
Jniv = Jniv';

% Grafica de contorno
printf("Pintando grafica de contorno... \n");
figure;
contour(vTheta0, vTheta1, Jniv, logspace(-2, 3, 20))
hold on;
plot(thetaJ(1), thetaJ(2), "rx", "MarkerSize", 8, "LineWidth", 4)
printf("Pulse Enter para continuar\n");
pause;
hold off

%Grafica surface
printf("Pintando grafica surface... \n");
figure;
surf(vTheta0, vTheta1, Jniv)
printf("Pulse Enter para finalizar\n");
pause;
