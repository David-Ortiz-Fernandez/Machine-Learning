function [JS , theta] = dGradiente(X, y, theta, alpha, iter)
    
m = length(y);
x = X(:,2);
cont =0 ;
%creamos un vector que almacene todos los valores de J para posteriormente 
  for iter = 1:iter  
          
      h = theta(1) + (theta(2)*x);
  
      theta0 = theta(1) - alpha * (1/m) * sum(h-y);
      theta1  = theta(2) - alpha * (1/m) * sum((h-y) .* x);
      
     %dejamos los valores de theta actualizados para utilizarlos en la iteración 
     %posterior  
      theta = [theta0; theta1];
      
      %En una matriz J se van almacenando los distintos valores de J para 
      %quedarnos el valor minimo al final
      JS(iter) = fCost(X, y, theta);  
      %%%%%%%%%%%%Modo depuración  
      %  hold on;
      %  if(cont <= 100) 
      %   cont++;
      %  else 
      %    cont = 0;
      %    plot(x, X*theta, "b-", "linewidth", 3)
      %    printf("Pulse Enter para continuar\n")
      %    pause;
      %  endif;  
   % hold off    

  endfor
endfunction