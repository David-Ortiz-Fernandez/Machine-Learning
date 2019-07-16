clear;
close all;
warning('off','all');
addpath("NN");
%cargamos los datos
data = csvread('TITANIC2.csv');

input_layer_size  = 8;   
hidden_layer1_size = 16;    
output_layer_size = 1; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================FASE 3:CROSS - VALIDATION===========================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataTr = data(1:785,:);
dataCv = data(786:1047,:);
dataTest = data(1048:end,:);

y_tr = dataTr(:,1);
X_tr = dataTr(:,2:end);
y_cv = dataCv(:,1);
X_cv = dataCv(:,2:end);
y_test = dataTest(:,1);
X_test = dataTest(:,2:end);


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

%PRECISION DE LA RED CON CROSS VALIDATION
printf('->Valor de lambda optimo: %f \n\n', bestlambda );
%entrenamiento con lambda elegido
params_rn = fmincg(@(t) (costeRN(t, input_layer_size, hidden_layer1_size, output_layer_size, X_tr, y_tr, bestlambda)), theta_inicial, opciones);
%porcentaje de aciertos en el test
[accu] = accuracyNN(params_rn, X_test, y_test, input_layer_size, hidden_layer1_size, output_layer_size);
printf('->Precision sobre datos de test: %f \n\n', accu * 100);
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



