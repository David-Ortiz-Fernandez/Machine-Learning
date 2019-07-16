function[lambda_vec,error_train,error_val]=validacion(X,y,Xval,yval,Xtest,ytest)
  %variables utiles
  lambda_vec = [ 0.01 0.03 0.1 0.3 1 3 10 30]';
  maxpercentage = 0;
  minerrorval = realmax;
  bestlambda = 0;

  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [m, n] = size(X);
  initial_theta = zeros(n , 1);

  for i = 1:length(lambda_vec)
      printf('Entrenando con lambda_vec %d/%d \n', i, length(lambda_vec));
      lambda = lambda_vec(i);
      %Para cada lambda entrenamos y calculamos los theta y el coste
      [theta,cost]=fminunc(@(t)(costeReg(t, X,y,lambda)),initial_theta,options);

%Calculamos el error con lambda 0, tiene que calcularse sin el termino 
%regularizado
    error_train(i) = costeReg(theta,X,y, 0);
    error_val(i) = costeReg(theta,Xval,yval,0);
 
      if (error_val(i) < minerrorval)
        minerrorval = error_val(i);
        bestlambda = lambda;
        besttheta = theta;
        bestcost = cost;
      endif
   endfor
   
   %imprimimos valores de error por pantalla
   printf('lambda\t\tTrain Error\tValidation Error\n');
      for i = 1:length(lambda_vec)
	      fprintf(' %f\t%f\t%f\n',lambda_vec(i), error_train(i), error_val(i));
     end
      
 %tras cros validation calculamos los aciertos para el test
    p = porcentaje(besttheta, Xtest, ytest);
    printf('El lambda optimo encontrado es %f que ha clasificado correctamente el %f de los datos de test. \n', bestlambda, p);
%pintado de seleccion de lambda
    figure(1);
    plot(lambda_vec, error_train, lambda_vec, error_val);
    legend('Entrenamiento', 'Validación');
    xlabel('lambda');
    ylabel('Error');
    printf('\n Presione Entrer para continuar.\n');
    pause;
%calculo de curvas de aprendizaje 
    for i = 1:m
        [theta, cost] = fminunc(@(t) (costeReg(t,X(1:i,:),y(1:i,:),bestlambda)),initial_theta, options);
        [jtrain(i), grad] = costeReg(theta, X(1:i,:), y(1:i,:), 0);
        [jval(i), grad] = costeReg(theta, Xval, yval, 0);
    endfor
%pintado de curva de aprendizaje
    figure(2);
    plot(1:m, jtrain, 1:m, jval);
    title('Curva de aprendizaje para la regresión logistica')
    legend('Entrenamiento', 'Validacion')
    xlabel('Numero de ejemplos de entrenamiento')
    ylabel('Error')
    
%RECALL Y PRECISION
    printf('FASE 4: Calculo del recall y precision. \n');
    %calculo del threshold con los datos de validacion
    th = threshold(besttheta, Xval, yval);
    %calculo de precision recall sobre datos de test
    [precision, recall] = precisionrecall(besttheta, Xtest, ytest, th);
    printf('Nuestro clasificador tiene una precision de %f y un recall de %f. El threshold optimo es de: %.2f. \n', precision * 100, recall * 100, th);
   
endfunction