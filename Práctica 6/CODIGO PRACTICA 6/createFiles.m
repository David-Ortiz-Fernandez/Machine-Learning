clear ; 
close all;

vocabList = getVocabList();
%flujo de creación de los dataset
%Añadimos ejemplos que son spam al dataset y los categorizamos como tal
    
%utilizando la función readFile incluida en la práctica para leer un fichero
%y devolver su contenido en una cadena y procesando el contenido del mensaje 
%con la función processEmail

for j = 1:500
    if (j > 0 && j < 10)
      file_contents = readFile(strcat('spam/000',num2str(j),'.txt'));
    elseif (j > 9 && j < 100)
      file_contents = readFile(strcat('spam/00',num2str(j),'.txt'));
    elseif (j > 99 && j <=500)
      file_contents = readFile(strcat('spam/0',num2str(j),'.txt'));
    endif

    email = processEmail(file_contents);
    
    email_splited = strsplit(email);
    email_as_vector = zeros(rows(vocabList), 1);
    email_as_vector = ismember(vocabList, email_splited);

    spam(j, :) = email_as_vector;
    printf(strcat(num2str(j),'/3301 \n'));
  endfor
%añadimos un 1 a cada fila para indicar que es spam    
    spam(:, columns(spam) + 1) = 1;
    save spam.mat;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 clear;
 vocabList = getVocabList();
%flujo de creación del dataset de entrenamiento
%Añadimos ejemplos que son spam al dataset y los categorizamos como tal
    
%utilizando la función readFile incluida en la práctica para leer un fichero
%y devolver su contenido en una cadena y procesando el contenido del mensaje 
%con la función processEmail 
 for j = 1:250
    if (j > 0 && j < 10)
      file_contents = readFile(strcat('hard_ham/000',num2str(j),'.txt'));
    elseif (j > 9 && j < 100)
      file_contents = readFile(strcat('hard_ham/00',num2str(j),'.txt'));
    elseif (j > 99 && j <= 250)
      file_contents = readFile(strcat('hard_ham/0',num2str(j),'.txt'));
    endif
  
 
    email  = processEmail(file_contents);
    
    email_sp = strsplit(email);
    email_vec = zeros(rows(vocabList), 1);
    email_vec = ismember(vocabList, email_sp);

    spam(j, :) = email_vec;
    printf(strcat(num2str(250 + j),'/3301 \n'));
  endfor
%añadimos a  cada fila un 1 indicando que es spam    
    hard_ham(:, columns(spam) + 1) = 1;
    save hard_ham.mat;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 clear;
 vocabList = getVocabList();  
%utilizando la función readFile incluida en la práctica para leer un fichero
%y devolver su contenido en una cadena y procesando el contenido del mensaje 
%con la función processEmail
 for j = 1:2551
    if (j > 0 && j < 10)
      file_contents = readFile(strcat('easy_ham/000',num2str(j),'.txt'));
    elseif (j > 9 && j < 100)
      file_contents = readFile(strcat('easy_ham/00',num2str(j),'.txt'));
    elseif (j > 99 && j < 1000)
      file_contents = readFile(strcat('easy_ham/0',num2str(j),'.txt'));
    elseif (j > 999 && j <= 2551)
      file_contents = readFile(strcat('easy_ham/',num2str(j),'.txt'));
    endif

    email = processEmail(file_contents);
    
    email_sp = strsplit(email);
    email_vec = zeros(rows(vocabList), 1);
    email_vec = ismember(vocabList, email_sp);

    easy_ham(j, :) = email_vec;
    printf(strcat(num2str(750 + j),'/3301 \n'));
  endfor
%añadimos a cada fila un 0 indicando que no es spam    
easy_ham(:, columns(easy_ham) + 1) = 0;
save easy_ham.mat;
printf('Presiones Enter para finalizar.\n');
pause;