function longrun_rate = rolling_longrun_AR1(X, y, j)
% y: T x 1 vector
% X: T x 2 matrix
% j: rolling window size, int
Tob = length(y);
W = Tob - j + 1;
longrun_rate = zeros(W,1);
for w=1:W
    foo_ols = OLS(X(w:(w+j-1), :), y(w:(w+j-1)));
    mu_hat = foo_ols(1);
    b_hat = foo_ols(2);
    longrun_rate(w) =  mu_hat / (1 - b_hat);
end

end

