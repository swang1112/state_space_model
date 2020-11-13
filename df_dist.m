clear
clc
%% H0: Random Walk
% time series lenth
Tob = 200;
% number of simulation
R = 5000;
e = randn(Tob, R);
y = cumsum(e, 1);

[DF_dist, alpha] = df_test(y);

Critical_val = quantile(DF_dist, [.01, .05, .10])
mean(alpha)
mean(DF_dist)
hist(DF_dist, 500)

%% Stationary AR(1)
u = randn(Tob, 1);
y_2 = zeros(Tob,1);
y_2(1) = 1;
for t = 2:Tob
    y_2(t) = 0.6*y_2(t-1) + u(t);
end

plot(y_2)
[df_2,  alpha_2] = df_test(y_2);
df_2 < Critical_val


