clear;
close all;
warning('off','all');

%las posiciones (i; j; 1); (i; j; 2) e (i; j; 3) representan respectivamente
%los porcentajes de rojo, verde y azul del pixel de la posicion (i,j)
A = double(imread('bird_small.png'));

%dividimos en 255 todos los pixeles para reresentar en rangos 
%entre 0 y 1
A = A / 255;
imagesc(A);

X = reshape(A, rows(A)  * columns(A), 3);

K = 32; 

randidx = randperm(size(X, 1));
centroides = X(randidx(1:K), :);
[centroides, idx] = runkMeans(X, centroides, 10, true);

  for i = 1:rows(X)
    X_comprimida(i, :) = centroides(idx(i),:);
  endfor

X_comprimida = reshape(X_comprimida,rows(A) ,columns(A), 3);
imagesc(X_comprimida);


