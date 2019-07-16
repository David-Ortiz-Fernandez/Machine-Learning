function [precision, recall] = precisionrecall(ytest,p)   
truepos = ytest + p == 2;
    if (p == 0)
        precision = 1;
    else
        precision = sum(truepos)/ sum(p);
    endif
    recall = sum(truepos) / sum(ytest);

    printf('Precision; %f,  Recall: %f \n', precision * 100, recall * 100);
endfunction




