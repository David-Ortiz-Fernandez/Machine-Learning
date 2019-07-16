function [precision, recall] = precisionrecall(params_rn,X,y,threshold, 
                                         num_entradas,num_ocultas,num_etiquetas)
   resultados = forwardpropagation(X, params_rn, num_entradas, num_ocultas, 
                                                                 num_etiquetas);
   %calculamos positivos reales en el test
    positivosReales = y == 1;
    positivosPred = resultados >= threshold;
        %almacenamos un 0 o un 1 dependiendo de si suman dos o no

    truepos = (positivosReales + positivosPred == 2);
  
  if (positivosPred == 0)
        precision = 1;
    else
        precision = sum(truepos) / sum(positivosPred);
    endif
    recall = sum(truepos) / sum(positivosReales);
endfunction


