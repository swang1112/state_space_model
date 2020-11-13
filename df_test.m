function [DF_stat, ar_coeff] = df_test(y)
% simple DF test
% y: matrix of dimension TxR
[T, R] = size(y);
DF_stat = zeros(R,1);
ar_coeff = zeros(R,1);
    for r=1:R
        [alpha_hat, alpha_se] = OLS(y(1:(T-1),r), y(2:T,r));
        
        ar_coeff(r) = alpha_hat;
        %ar_coeff(r) = inv(y(1:(T-1),r)'*y(1:(T-1),r)) * y(1:(T-1),r)'*y(2:T,r);
        %SSR = y(2:T,r)'*y(2:T,r) - ar_coeff(r) * y(1:(T-1),r)'*y(2:T,r);
        %sigma2 = SSR/ (T-1);
        %alpha_se = sqrt(sigma2) / (y(1:(T-1),r)'*y(1:(T-1),r));
        DF_stat(r) = (ar_coeff(r) - 1) / alpha_se;
    end
end

