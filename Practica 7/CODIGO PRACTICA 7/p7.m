clear;
close all;
warning('off','all');

load('ex7data2.mat');

K = 3; % 3 Centroides
inicial_centroides = [3 3; 6 2; 8 5];
idx = findClosestCentroids(X, inicial_centroides);

centroids = computeCentroids(X, idx, K);

randidx = randperm(size(X, 1));
centroides = X(randidx(1:K), :);
[centroids, idx] = runkMeans(X, centroides, 10, true);
