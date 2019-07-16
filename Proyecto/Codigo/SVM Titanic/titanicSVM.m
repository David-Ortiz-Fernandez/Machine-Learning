clear;
close all;
warning('off','all');
addpath("SVM");
%cargamos los datos
data = csvread('data.csv');

%1309 totales, el 60% es 785, el 20% es 262
dataTr = data(1:785,:);
dataCV = data(786:1047,:);
dataTest = data(1048:end,:);
dta=data(1:end,:);

y= dta(:,1);
X= dta(:,3:end);
y_cv = dataCV(:,1);
X_cv = dataCV(:,3:end);
y_tr = dataTr(:,1);
X_tr = dataTr(:,3:end);
y_test = dataTest(:,1);
X_test = dataTest(:,3:end);

%posibles valores utiles
[m , n]=size(X);
[m_tr , n_tr]=size(X_tr);
[m_cv , n_cv] = size(X_cv);
[m_test , n_test]=size(X_test);

%NORMALIZACION 
%[X_tr,mu,sigma] = normalizaAtributo(X_tr);
%[X_test,mu,sigma] = normalizaAtributo(X_test);
%[X_cv,mu,sigma] = normalizaAtributo(X_cv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=================================FASE 1=======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%entrenamiento con kernell lineal
printf('=====FASE 1: SVM CON KERNELL LINEAL=====\n \n');
C = 1;
%====================================KERNELL LINEAL
tic;
model = svmTrain(X_tr, y_tr, C, @linearKernel, 1e-3, 20);
time = toc;
printf('->Se ha calculado el modelo con un maximo de 20 iteraciones y un valor de C = 1 \n')
printf('->Duracion del proceso %.2f segundos\n', time);

%calcula los aciertos con el total de los entrenados y un kernell lineal %
p = svmPredict(model,X_tr); %
printf('->Porcentaje de acierto con kernell linel entrenando y prediciendo con los datos de entrenamiento: %f\n', mean(double(p == y_tr)) * 100);
printf('Pulse Enter para continuar...\n');
pause;
%calcula los aciertos sobre los datos de test y un kernell lineal %
p = svmPredict(model,X_test); %
printf('->Porcentaje de acierto con kernell linel prediciendo sobre los datos de test: %f\n', mean(double(p == y_test)) * 100);
printf('Pulse Enter para continuar...\n');
pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=================================FASE 2=======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%====================================KERNELL GAUSIANO
printf('=====FASE 2: ENTRENAMIENTO DEL SVM CON KERNELL GAUSIANO\n');
C = 1; 
sigma = 0.1;

tic;
model = svmTrain(X_tr, y_tr, C, @(x1,x2) gaussianKernel(x1,x2,sigma));
time = toc;
printf('->Se ha calculado el modelo con un maximo de 20 iteraciones y un valor de C = 1 y sigma=0.1 \n')
printf('->Duracion del proceso: %f segundos. \n', time);

%calculo del accuracy con el total entrenado
p = svmPredict(model, X_tr);
printf('Porcentaje de acierto del modelo sobre los datos de entrenamiento: %f\n', mean(double(p == y_tr)) * 100);
printf('Pulse Enter para continuar...\n');

%calculo del accuracy con el total entrenado
p = svmPredict(model, X_test);
printf('Porcentaje de acierto del modelo sobre los datos de test: %f\n',mean(double(p == y_test)) * 100);
printf('Pulse Enter para continuar...\n');
pause;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=================================FASE 3=======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%entrenamiento con kernell gausiano
printf('=====FASE 3: ENTRENAMIENTO DEL SVM CON KERNELLGAUSIANO CSIG. OPTIMOS\n');
%Elección de los parámetros C y sigma optimos
[C, sigma] = calculoCSigma(X_tr, y_tr, X_cv, y_cv);

printf('->Valores para C y para sigma optimos son %f y %f.\n\n', C, sigma);
%entrenamos el modelo con estos valores devueltos en la funcion anterior
model= svmTrain(X_tr, y_tr, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

p = svmPredict(model, X_test);
printf('Porcentaje de acierto del modelo entrenado con kernell gausiano y los valores C y sigma optimos: %f\n', mean(double(p == y_test)) * 100);
printf('Pulse Enter para continuar...\n');
pause;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=================================FASE 4=======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf('=====FASE 4: RECALL Y PRECISION\n');
precisionrecall(y_test,p);

  