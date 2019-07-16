%función sigmoide
function g = fsigmoide(z)

g = 1.0 ./ (1.0 + exp(-z));

endfunction
