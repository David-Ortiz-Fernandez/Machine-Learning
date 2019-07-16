function[acc]= accuracyNN(theta, X, y, num_entradas, num_ocultas, num_etiquetas)
    m=size(X,1);
    num_fallos = 0;
    p = zeros(size(X, 1), 1);
    
    h = forwardpropagation(X, theta, num_entradas, num_ocultas, num_etiquetas);
    [probs, p] = max(h,[],2);
   
    p = p .- 1;
    acc=(mean(double(p == y))) ;
    
endfunction


