%funcion para el calculo de los valores de C y sigma y maximizan los valores 
%bien clasificados
function [C, sigma] = calculoCSigma(X, y, Xval, yval)
minerror = inf;
values = [0.01 0.03 0.1 0.3 1 3 10 30];

%doble for para los 64 modelos diferentes
for Caux = values
  for sigmaAux = values
    printf('Entrenamiento y validacion para los valores de: \n[Caux, sigmaAux] = [%f %f]\n', Caux, sigmaAux);
   
   %culando el porcentaje de estos ejemplos que clasificaca correctamente
    model = svmTrain(X, y, Caux, @(x1, x2) gaussianKernel(x1, x2, sigmaAux));
   
   %calculo de los que se clasifican mal
    error = mean(double(svmPredict(model, Xval) ~= yval));
    printf('Prediccion de error: %f\n', error);
    
    %actualizacion del error minimo en caso de haber encontado uno menor
    if( error <= minerror )
    sigma = sigmaAux;
    C = Caux;
    minerror = error;
    endif
    

  endfor
endfor
    printf('Valores de  C = %f , sigma = %f con un minimo error de prediccion de:  %f\n', C, sigma, minerror);
    printf('--------------------------------------------------------------\n');
endfunction
