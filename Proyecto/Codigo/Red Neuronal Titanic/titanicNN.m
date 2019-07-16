clear;
close all;
warning('off','all');
addpath("NN");
%cargamos los datos
data = csvread('data.csv');
%dividimos las features y los label y extraemos la feature ID(columna 2)
dta=data(1:end,:);
%no le añadimos aun los 1´s porque posteiormente se añadirán los bias unit
y= dta(:,1);
X= dta(:,3:end);

%posibles valores útiles
[m , n]=size(X);
%normalización
%[X,mu,sigma] = normalizaAtributo(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=================================FASE 1=======================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============= Inicializacion red Neuronal ===================================
%tamaño de las capas
input_layer_size  = 7;   
hidden_layer1_size = 9;    
output_layer_size = 1; 

printf('=====ENTRENAMIENTO DE LA RED NEURONAL DE UNA CAPA OCULTA DE 9N=====\n \n');
printf('1-ENTRENAMIENTO DE LA RED NEURONAL CON EL TOTAL DEL DATASET\n \n');

%pesos iniciales aleatorios 
initial_theta1 = pesosAleatorios(input_layer_size,hidden_layer1_size);
initial_theta2 = pesosAleatorios(hidden_layer1_size ,output_layer_size);

%enrrollado de parámetros
theta_inicial = [initial_theta1(:) ; initial_theta2(:)];
   
opciones = optimset('MaxIter', 100);   
lambda = 3;

%chequeo retropropagación correcta con derivadas parciales
checkNNGradients(lambda);

%% ============= Entrenamiento red neuronal con el total de los datos ==========
tic;
params_rn = fmincg(@(t) (costeRN(t, input_layer_size, hidden_layer1_size, output_layer_size, X, y, lambda)), theta_inicial, opciones);
time = toc;
printf('->Se han obtenido los pesos optimos con un maximo de 100 iteraciones y un parametro de regularizacion lambda = 3 \n');
[cost, grad] = costeRN(params_rn, input_layer_size, hidden_layer1_size, output_layer_size, X, y, lambda);

printf('->El entrenamiento ha durado: %0.2f segundos y el coste minimo ha sido: %f. \n', time, cost);
printf('Pulse Enter para continuar...\n \n');
pause();

%% ===============Accuracy con el total de los datos ===========================
[accu] = accuracyNN(params_rn, X, y, input_layer_size, hidden_layer1_size, output_layer_size);
printf('->Precision entrenando con el total del dataset: %f \n\n', accu * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%================================FASE 2========================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf('2-ENTRENAMIENTO DE LA RED NEURONAL CON EL 70 POR CIENTO DEL DATASET\n');
dataTr = data(1:916,:);
dataTest = data(917:end,:);

y_tr = dataTr(:,1);
X_tr = dataTr(:,3:end);
y_test = dataTest(:,1);
X_test = dataTest(:,3:end);

%posibles valores utiles
[m_tr , n_tr]=size(X_tr);
[m_test , n_test]=size(X_test);
%Normalización de atributos
%[X_tr,mu,sigma] = normalizaAtributo(X_tr);
%[X_test,mu,sigma] = normalizaAtributo(X_test);

%pesos iniciales aleatorios 
initial_theta1 = pesosAleatorios(input_layer_size,hidden_layer1_size);
initial_theta2 = pesosAleatorios(hidden_layer1_size ,output_layer_size);
%enrrollado de parámetros
theta_inicial = [initial_theta1(:) ; initial_theta2(:)];
opciones = optimset('MaxIter', 100);   
lambda = 3;

%% ============= Entrenamiento red neuronal con el 70% de los datos ==========
tic;
params_rn = fmincg(@(t) (costeRN(t, input_layer_size, hidden_layer1_size, output_layer_size, X_tr, y_tr, lambda)), theta_inicial, opciones);
time = toc;
printf('->Se han obtenido los pesos optimos con un maximo de 100 iteraciones y un parametro de regularizacion lambda = 3 \n');
[cost, grad] = costeRN(params_rn, input_layer_size, hidden_layer1_size, output_layer_size, X_tr, y_tr, lambda);

printf('->El entrenamiento ha durado: %0.2f segundos y el coste minimo ha sido: %f. \n', time, cost);
printf('Pulse Enter para continuar...\n \n');
pause(); 
[accu] = accuracyNN(params_rn, X_test, y_test, input_layer_size, hidden_layer1_size, output_layer_size);
printf('->orcentaje de acierto entrenando con el 70 por ciento del dataset, sobre el 30 por ciento restante destinado a test: %f \n\n', accu * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================FASE 3:CROSS - VALIDATION===========================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataTr = data(1:785,:);
dataCv = data(786:1047,:);
dataTest = data(1048:end,:);

y_tr = dataTr(:,1);
X_tr = dataTr(:,3:end);
y_cv = dataCv(:,1);
X_cv = dataCv(:,3:end);
y_test = dataTest(:,1);
X_test = dataTest(:,3:end);

%posibles valores utiles
[m_tr , n_tr]=size(X_tr);
[m_cv , n_cv]=size(X_cv);
[m_test , n_test]=size(X_test);
%crea un vector lambda
lambda_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10 30]';
minCost = realmax;

%normalizacion de los atributos
%[X_tr,mu,sigma] = normalizaAtributo(X_tr);
%[X_cv,mu,sigma] = normalizaAtributo(X_cv);
%[X_test,mu,sigma] = normalizaAtributo(X_test);

%pesos iniciales aleatorios 
initial_theta1 = pesosAleatorios(input_layer_size,hidden_layer1_size);
initial_theta2 = pesosAleatorios(hidden_layer1_size ,output_layer_size);

%enrrollado de parámetros
theta_inicial = [initial_theta1(:) ; initial_theta2(:)];
opciones = optimset('MaxIter', 100);   

%seleccion de lambda
for i = 1:rows(lambda_vec)        
    printf('Entrenando con lambda: %d \n',lambda_vec(i));
    %entrenamiento
    params_rn = fmincg(@(t) (costeRN(t, input_layer_size, hidden_layer1_size, output_layer_size, X_tr, y_tr, lambda_vec(i))), theta_inicial, opciones);
    %guardamos los valores para posteriormente construir las gráficas
    [jtrain(i), grad] = costeRN(params_rn, input_layer_size, hidden_layer1_size, output_layer_size, X_tr, y_tr, 0); 
    [jval(i), grad] = costeRN(params_rn, input_layer_size, hidden_layer1_size, output_layer_size, X_cv, y_cv, 0);
       
    if(jval(i) < minCost)
       minCost = jval(i);
       minCostCv = jtrain(i);
       bestlambda = lambda_vec(i);
       bestparms_rn = params_rn;
     endif
endfor

%PLOT de seleccion de lambda
figure(1);
plot(lambda_vec, jtrain, lambda_vec, jval);
legend('Entrenamiento', 'Validación');
xlabel('lambda');
ylabel('Error');
printf('\n Presione Entrer para continuar.\n');
pause;
%valores de error y lambda por pantalla
printf('lambda\t\tTrain Error\tValidation Error\n');
  for i = 1:length(lambda_vec)
	   fprintf(' %f\t%f\t%f\n',lambda_vec(i), jtrain(i), jval(i));
   end

%PRECISION DE LA RED CON CROSS VALIDATION
printf('->Valor de lambda optimo: %f \n\n', bestlambda );
%entrenamiento con lambda elegido
params_rn = fmincg(@(t) (costeRN(t, input_layer_size, hidden_layer1_size, output_layer_size, X_tr, y_tr, bestlambda)), theta_inicial, opciones);
%porcentaje de aciertos en el test
[accu] = accuracyNN(params_rn, X_test, y_test, input_layer_size, hidden_layer1_size, output_layer_size);
printf('->Porcentaje de aciertos sobre datos de test: %f \n\n', accu * 100);
pause;


%RECALL Y PRECISION
printf('FASE 4: Calculo del recall y precision. \n');
%calculo del threshold con los datos de validacion
th = threshold(bestparms_rn, X_cv, y_cv, input_layer_size, hidden_layer1_size, output_layer_size);
%calculo de precision recall sobre datos de test
[precision, recall] = precisionrecall(bestparms_rn, X_test, y_test, th, input_layer_size, hidden_layer1_size, output_layer_size);
printf('Nuestro clasificador tiene una precision de %f y un recall de %f. El threshold optimo es de: %.2f. \n', precision * 100, recall * 100, th);
   

%========================CURVA DE APRENDIZAJE===================================
%recordar que para las curvas de aprendizaje es necesario que el parametro de 
%Importante recordar regularización sea 0 para no incluirlo
lambda = 0;
for i = 1:m_tr
%entrenamos
 params_rn = fmincg(@(t) (costeRN(t, input_layer_size, hidden_layer1_size, 
output_layer_size, X_tr(1:i,:), y_tr(1:i,:), bestlambda)), theta_inicial, opciones);
%almacenamos los costes       
  [jtrain(i), grad] = costeRN(params_rn, input_layer_size, hidden_layer1_size, 
                         output_layer_size, X_tr(1:i,:), y_tr(1:i,:), 0);
 
  [jval(i), grad] = costeRN(params_rn, input_layer_size, hidden_layer1_size, 
                                        output_layer_size, X_cv, y_cv, 0); 
endfor

%pintado de curva de aprendizaje
figure(2);
plot(1:m_tr, jtrain, 1:m_tr, jval);
title('Curva de aprendizaje para la red neuronal')
legend('Entrenamiento', 'Validacion')
xlabel('Numero de ejemplos de entrenamiento')
ylabel('Error')

printf('\n Pulse Entrer para finalizar.\n');
pause;

%RECALL Y PRECISION
printf('FASE 4: Calculo del recall y precision. \n');
%calculo del threshold con los datos de validacion
th = threshold(bestparms_rn, X_cv, y_cv, input_layer_size, hidden_layer1_size, output_layer_size);
%calculo de precision recall sobre datos de test
[precision, recall] = precisionrecall(bestparms_rn, X_test, y_test, th, input_layer_size, hidden_layer1_size, output_layer_size);
printf('Nuestro clasificador tiene una precision de %f y un recall de %f. El threshold optimo es de: %.2f. \n', precision * 100, recall * 100, th);
   
