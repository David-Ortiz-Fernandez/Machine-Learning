function I = mcint(fun, a, b, num_puntos)
  x=[a:0.1:b];
  nDebajo=0;
  nTotal=num_puntos;
  
  M=max(fun(x));

puntosx=[];
puntosy=[];

tic();
  for i=1:num_puntos
    
    x1=rand();
    y1=rand();
    puntosx=[puntosx x1];
    puntosy=[puntosy y1];
   if ((x1>a)&&(x1<b)&&(y1<fun(x1))) 
     nDebajo++;
      endif
endfor

 add_time = toc()

  plot(x, fun(x), "b-", "markersize", 4, "linewidth", 2,puntosx,puntosy,"rx","markersize", 4, "linewidth", 2)
  xlabel("x", "fontsize", 20)
  ylabel("fun(x)", "fontsize", 20)
  legend("fun(x)")
  grid on

  nDebajo
  num_puntos
  b
  a
  M
  
  
I=(nDebajo/num_puntos)*(b-a)*M;
endfunction
