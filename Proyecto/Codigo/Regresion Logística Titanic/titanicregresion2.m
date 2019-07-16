clear ; 
close all ;
addpath("regresion");
warning('off','all');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============ REGRESION LOGISTICA REGULARIZADA CON CROSS VALIDATION =========%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf('FASE 4: entrenamiento con el 60 aplicando cross- validation. \n');
%cargamos los datos
data = csvread('TITANIC2.csv');
%dividimos las features y los label
dataTr = data(1:785,:);
dataCV = data(786:1047,:);
dataTest = data(1048:end,:);
% sacamos el ID de los datos y separamos las features de los label (column 2) 
y_cv = dataCV(:,1);
X_cv = dataCV(:,2:end);
y_tr = dataTr(:,1);
X_tr = dataTr(:,2:end);
y_test = dataTest(:,1);
X_test = dataTest(:,2:end);
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
