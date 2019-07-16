function [X_norm, mu, sigma] = normalizaAtributo(X)

 X_norm=X;
 
  for i = 1:size(X,2) %una iteracción para cada atributo
      mu(i) = mean(X(:,i));
      sigma(i) = std(X(:,i));
      
      X_norm(:,i) = (X_norm(:,i)-mu(i))/sigma(i);
  endfor

end


%function [X_norm, maxim] = featureNormalize(X)

 %   maxim = max (X);
 %   X_norm = X ./ maxim;

%end