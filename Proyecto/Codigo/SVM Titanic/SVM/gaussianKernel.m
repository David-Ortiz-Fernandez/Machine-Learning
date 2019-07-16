%funci�n que calcule el kernel gaussiano para as� poder entrenar una SVM que 
%clasifique correctamente el segundo conjunto de datos
function sim = gaussianKernel(x1, x2, sigma)
x1 = x1(:); x2 = x2(:);

%la funci�n de kernel gaussiano calcula la distancia entre dos ejemplos de 
%entrenamiento (x(i), x(j))
sim = 0;
sim = exp(-1*(x1-x2)'*(x1-x2)/(2*sigma*sigma));

endfunction

