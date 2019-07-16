function I = mcint2(fun, a, b, num_puntos)
  x=[a:0.1:b];
  M=max(fun(x));
  
  X1=rand(1,num_puntos);
  Y1=rand(1,num_puntos);
  
  tic();  
  R=find(Y1 < fun(X1)); 
  nDebajo=columns(R);
  time = toc()
 

 
  plot(x, fun(x), "b-", "markersize", 4, "linewidth", 2,X1,Y1,"rx","markersize", 4, "linewidth", 2)
  xlabel("x", "fontsize", 20)
  ylabel("fun(x)", "fontsize", 20)
  legend("fun(x)")
  grid on
  
  nDebajo
  num_puntos
  b
  a
  M
  
  I=(nDebajo/num_puntos)*(b-a)*M
endfunction
