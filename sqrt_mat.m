X = randn(1000,3);

Sigma = rand(3,3)+1;
Sigma = Sigma'* Sigma;

% how to compute sqrt of matrix
% 1. chol
Sigma_sqrt_chol = chol(Sigma);
X_1 =  X * Sigma_sqrt_chol;
cov(X_1);
Sigma;

% 2. eig
[Sigma_vec, Sigma_val] = eig(Sigma);
X_2 = X * Sigma_vec * sqrt(Sigma_val);
cov(X_2);
Sigma;

% 3. SVD
[U,S,V] = svd(Sigma);
X_3 = X * U * sqrt(S)*V';
cov(X_3)
Sigma
