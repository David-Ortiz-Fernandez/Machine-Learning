clear ; 
close all;
%1. Flujo de datos:Support Vector Machines
%Visualizamos los datos
load('ex6data1.mat');

% Plot datos ex6data1.mat
% Plot para apreciar que los datos son linealmente separables
plotData(X, y);

printf('Presione Enter para continuar.\n');
pause;
%1.1. Kernel lineal
    %Probar con C=1 y C=100
    C = 1;
    %C = 100;
    model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
    %visualizar la frontera de separación
    visualizeBoundaryLinear(X, y, model);
    
    p = svmPredict(model, X);
    printf('Porcentaje de acierto del modelo: %f\n', mean(double(p == y)) * 100);
    
    printf('Presione Enter para continuar.\n');
    pause;

%1.2. Kernel gaussiano
    load('ex6data2.mat');

    % Plot para apreciar que los datos no son linealmente separables
    % Plot datos ex6data2.mat
    plotData(X, y);

    C = 1; sigma = 0.1;
    
    %entrenamiento del modelo
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    %visualizar la frontera de separación
    visualizeBoundary(X, y, model);

    p = svmPredict(model, X);
    printf('Porcentaje de acierto del modelo: %f\n', mean(double(p == y)) * 100);
    
    fprintf('Presione Enter para continuar.\n');
    pause;

%1.3. Elección de los parámetros C y s
    load('ex6data3.mat');
    % Plot ex6data3.mat
    plotData(X, y);

    printf('Presione Enter para continuar.\n');
    pause;

    %calculo de los parametros C y sigma
    [C, sigma] = calculoCSigma(X, y, Xval, yval);

    %entrenamiento del modelo
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    visualizeBoundary(X, y, model);
    
    p = svmPredict(model, X);
    printf('Porcentaje de acierto del modelo: %f\n', mean(double(p == y)) * 100);