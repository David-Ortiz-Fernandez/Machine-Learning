function th = threshold(theta, X, y)
  candidato = [0 : 0.01 : 1];
  maxf1 = 0;
  %para candidato de ser el threshold optimo
    for i = 1:columns(candidato)
    %calculamos para cada candidato a ser el threshold su precision y su recall
       [precision, recall] = precisionrecall(theta, X, y, candidato(:,i));            
       f1score = 2 * ((precision * recall) / (precision + recall));
    %elegimos el de mayor score
       if(maxf1 < f1score)
          maxf1 = f1score;
          th = candidato(:,i);
        endif
    endfor
endfunction

