function centroids = computeCentroids(X, idx, K)
[m n] = size(X);

for i = 1:K
 points = idx == i;
 suma = zeros(1,columns(X) );
    for j = 1:rows(points)
      if(points(j, :) == 1) 
        suma = suma + X(j, :);
      endif
    endfor
    centroids(i, :) = suma./sum(points);
  endfor
endfunction

