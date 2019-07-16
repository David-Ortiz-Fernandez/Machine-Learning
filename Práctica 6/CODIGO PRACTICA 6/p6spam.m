clear ; 
close all;
warning('off','all');

%Cargamos easy_ham
load('easy_ham.mat');
[m_easy_ham,n] = size(easy_ham);
  
%Cargamos hard_ham
load('hard_ham.mat');
[m_hard_ham,n] = size(hard_ham);
  
%Cargamos spam
load('spam.mat');
[m_spam,n_spam] = size(spam);
  
%elegimos porcentajes del 60%,20% y 20% de CADA archivo
datatr_easy_ham = easy_ham(1:floor(0.6*m_easy_ham),:);
datacv_easy_ham = easy_ham(floor(0.6*m_easy_ham)+1:floor(0.8*m_easy_ham),:);
datatest_easy_ham = easy_ham(floor(0.8*m_easy_ham)+1:end,:);

datatr_hard_ham = hard_ham(1:floor(0.6*m_hard_ham),:);
datacv_hard_ham = hard_ham(floor(0.6*m_hard_ham)+1:floor(0.8*m_hard_ham),:);
datatest_hard_ham = hard_ham(floor(0.8*m_hard_ham)+1:end,:);

datatr_spam = spam(1:floor(0.6*m_spam),:);
datacv_spam = spam(floor(0.6*m_spam)+1:floor(0.8*m_spam),:);
datatest_spam = spam(floor(0.8*m_spam)+1:end,:);

X_tr = [datatr_easy_ham ; datatr_hard_ham ; datatr_spam];
%y es la ultima fila de X
y_tr = X_tr(:, end);
%quitamos y de X
X_tr=X_tr(:,1:end-1);
X_cv = [datacv_easy_ham ; datacv_hard_ham ; datacv_spam];
%y es la ultima fila de X
y_cv = X_cv(:, end);
%quitamos y de X
X_cv=X_cv(:,1:end-1);
X_test = [datatest_easy_ham ; datatest_hard_ham ; datatest_spam];
%y es la ultima fila de X
y_test = X_test(:, end);
%quitamos y de X
X_test=X_test(:,1:end-1);
C = 0.1;
model = svmTrain(X_tr, y_tr, C, @linearKernel);
%Evaluacion del modelo
p = svmPredict(model, X_test);
printf('Porcentaje de acierto del modelo con kernel linial: %f\n', 
                                              mean(double(p == y_test)) * 100);   
%entrenamiento con kernell gausiano
C = 1; 
sigma = 0.1;
[C, sigma] = calculoCSigma(X_tr, y_tr, X_cv, y_cv);
model= svmTrain(X_tr, y_tr, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
%evaluacion del modelo
p = svmPredict(model, X_tr);
printf('Porcentaje de acierto del modelo con kernel gausiano: %f\n', mean(double(p == y_tr)) * 100);