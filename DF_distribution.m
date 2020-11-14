clear
clc
%% H0: Random Walk
% time series lenth
Tob = 200;
% number of simulation
R = 5000;
e = randn(Tob, R);
y = cumsum(e, 1);

[DF_dist, alpha] = df(y);

Critical_val = quantile(DF_dist, [.01, .05, .10])
mean(alpha)
hist(DF_dist, 500)
for cv=1:3
    line([Critical_val(cv), Critical_val(cv)], [0, max(histcounts(DF_dist, 500))], 'Color', 'r', 'Linewidth', 1.5)
end 
title("Dickey-Fuller Distribution and critical values for unit root test")
%% Stationary AR(1)
u = randn(Tob, 1);
y_2 = zeros(Tob,1);
y_2(1) = 1;
for t = 2:Tob
    y_2(t) = 0.9*y_2(t-1) + u(t);
end

[df_2,  alpha_2] = df(y_2);
df_2 < Critical_val


