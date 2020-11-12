function [beta_hat, beta_se] = OLS(X, y)
% arg:
% size of X: TxK
% size of y: Tx1
[T, K] = size(X);
beta_hat = inv(X'*X) * X'* y;
SSR = y'* y - beta_hat' * X' * y;
sigma_hat = SSR / (T - K);
beta_se = sqrt(sigma_hat) * diag(inv(X'*X));
end
