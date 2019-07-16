function [precision, recall] = precisionrecall(theta, Xtest, ytest, threshold)

    resultados = fsigmoide(Xtest*theta);
    %calculamos positivos reales en el test
    positivosReales = ytest == 1;
    
    positivosPred = resultados >= threshold;
    %alamacenamos un 0 o un 1 dependiendo de si suman dos o no
    truepos = (positivosReales + positivosPred == 2);
    
    if (positivosPred == 0)
        precision = 1;
    else
        %calculo de la precision
        precision = sum(truepos) / sum(positivosPred);
    endif
    %calculo del recall
    recall = sum(truepos) / sum(positivosReales);
endfunction


