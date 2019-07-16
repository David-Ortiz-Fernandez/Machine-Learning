function idx = findClosestCentroids(X, centroids)

%bucle para cada centroide
for i=1:rows(X)
  normaMin = norma(X(i, :), centroids(1, :)) ^ 2;
    idx(i,:) = 1;
    for j=2:rows(centroids)
      c = norma(X(i, :), centroids(j, :));
      if (c < normaMin)
        idx(i,:) = j;
        normaMin = c;
      endif
    end
    %devolvemos el minimo
end
end



