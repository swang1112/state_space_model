function[beta] = N_post(x, y, var_e, b0, B0)

be = inv(x'*x)*x'*y; 

sample_estimate_Gewicht =  inv(var_e)*(x'*x); 
prior_Gewicht = inv(B0); 

b1 = inv(inv(B0) + inv(var_e)*x'*x)*(inv(B0)*b0 + inv(var_e)*(x'*x)*be);
B1  = inv(inv(B0) + inv(var_e)*x'*x);

beta = b1 + chol(B1)*randn(size(b1,1),1); 

end