function th = threshold(theta, X, y, num_entradas, num_ocultas, num_etiquetas)
 %vector de candidatos  
 candidato = [0 : 0.01 : 1]';
 maxf1 = 0;

for i = 1:rows(candidato)
  [precision, recall] = precisionrecall(theta, X, y, candidato(i),num_entradas, 
                                                   num_ocultas, num_etiquetas);
   f1score = 2 * ((precision * recall) / (precision + recall));

        if(maxf1 < f1score)
            maxf1 = f1score;
            th = candidato(i);
        endif
    endfor
endfunction

