clear ; 
close all ;
addpath("regresion");
warning('off','all');
%cargamos los datos del csv
data = csvread('data.csv');
% sacamos la feature ID de los datos y separamos las features de los label 
%LEGENDA [Survived, PassengerId, Pclass, Sex, Age, SibSp, Parch, 
%                Fare, Cabin, Embarked (S=1, C=2 , Q=3)] 
y = data(:,1);
X = data(:,3:end);
[m , n]=size(X);
%% ============ REGRESION LOGISTICA SOBRE EL 100% DEL DATASET===================
%añadimos columna de 1´s a las matrices
X = [ones(m, 1) X];

%A continuación, hemos aplicado regresión logística  sobre el 100% de los datos
printf('FASE 1: regresión logi­stica regularizada sobre el 100 del dataset. \n');
printf('Pulsa Enter para continuar...\n');
pause;
%inicializamos los valore theta a 0
thetainicial = zeros(n + 1, 1);
lambda = 0;
opciones = optimset('GradObj', 'on', 'MaxIter', 400);

%Normalizacion de atributos
%[X,mu,sigma] = normalizaAtributo(X);
% Coste y gradiente con theta iniciales
[cost, grad] = costeReg(thetainicial, X, y, lambda);
printf('->Función J de Coste para valores theta iniciales: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;

%Aplicacion del descenso de gradiente con fminunc
lambda = 2;
tic;
% Obtención de los valores óptimos de theta
[theta, cost] = fminunc(@(t)(costeReg(t, X, y, lambda)),thetainicial, opciones);
time = toc;

printf('->Coste minimo con valores lambda de 2 y MAXITER de 400 : %f\n', cost);
printf('->El tiempo para alcanzar convergencia a sido: %.2f segundos\n',time);

printf('->Porcentaje de aciertos sobre el total del dataset, tras entrenar con todo.');
p = porcentaje(theta, X, y);
printf('Presione Enter para continuar.\n');
pause;
printf('===================================================================\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============ REGRESION LOGISTICA SOBRE EL 70% DEL DATASET ==================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf('FASE 2: regresión logi­stica regularizada sobre el 70 del dataset. \n');
printf('Pulsa Enter para continuar...\n');
dataTr = data(1:916,:);
dataTest = data(917:end,:);
y_tr = dataTr(:,1);
X_tr = dataTr(:,3:end);
y_test = dataTest(:,1);
X_test = dataTest(:,3:end);
%valores utiles
[m_tr , n_tr]=size(X_tr);
[m_test , n_test]=size(X_test);

%añadimos columna de 1´s a las matrices para theta 0
X_tr = [ones(m_tr, 1) X_tr];
X_test = [ones(m_test, 1) X_test];

%inicializamos los valore theta 0 y lambda a 2
thetainicial = zeros(n_tr + 1, 1);
lambda = 3;

%Cálculo del valor óptimo de los parámetros con fminunc y theta optimas
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
tic;
[theta, cost]=fminunc(@(t)(costeReg(t,X_tr,y_tr,lambda)),thetainicial,opciones);
tiemporeg = toc;
printf('Precisión calculada tras entrenar el modelo con el 70 por ciento, y tetear con el 30 por ciento');
 p = porcentaje(theta, X_test, y_test);
 pause;
printf('===================================================================\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============ REGRESION LOGISTICA REGULARIZADA CON CROSS VALIDATION =========%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf('FASE 3: entrenamiento con el 60 aplicando cross- validation. \n');
%cargamos los datos
data = csvread('data.csv');
%dividimos las features y los label y extraemos la feature ID
dataTr = data(1:785,:);
dataCV = data(786:1047,:);
dataTest = data(1048:end,:);
% sacamos el ID de los datos y separamos las features de los label (column 2) 
y_cv = dataCV(:,1);
X_cv = dataCV(:,3:end);
y_tr = dataTr(:,1);
X_tr = dataTr(:,3:end);
y_test = dataTest(:,1);
X_test = dataTest(:,3:end);
%valores utiles
[m_tr , n_tr]=size(X_tr);
[m_cv , n_cv] = size(X_cv);
[m_test , n_test]=size(X_test);

%añadimos columna de 1´s a las matrices para theta 0
X_tr = [ones(m_tr, 1) X_tr];
X_cv = [ones(m_cv, 1) X_cv];
X_test = [ones(m_test, 1) X_test];

%inicializamos los valore theta 0 y lambda a 3
thetainicial = zeros(n_tr + 1, 1);
lambda = 3;

% Coste inicial y gradiente sobre el 60% de los datos
[cost, grad] = costeReg(thetainicial, X_tr, y_tr, lambda);
printf('->Coste para valores theta iniciales: %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;

%Cálculo del valor óptimo de los parámetros con fminunc y theta optimas
opciones = optimset('GradObj', 'on', 'MaxIter', 400);
tic;
[theta, cost]=fminunc(@(t)(costeReg(t,X_tr,y_tr,lambda)),thetainicial,opciones);
tiemporeg = toc;

printf('->Coste minimo con valores lambda de 2 y MAXITER de 400 : %f\n', cost);
printf('Presione Enter para continuar.\n');
pause;

%Curvas de validacion y aplicacion de cross validation
[lambda_vec,error_train,error_val]=validacion(X_tr,y_tr,X_cv,y_cv,X_test,y_test);
printf('Presione Enter para finalizar.\n');
pause;


%dataTr = data(1:535,:);
%dataCV = data(536:714,:);
%dataTest = data(715:891,:);