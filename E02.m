%_____%
% E01 %
%_____%

clc
clear
%% 1.
X = randn(100, 1);
u = randn(100, 1);
beta = 5;
y = beta * X + u;
beta_hat = inv(X'*X) * X'* y;
y_hat = beta_hat *  X;
u_hat = y - y_hat;

% plots
subplot(2,2,1);
plot(1:100, u);
title("$$u$$", 'Interpreter','Latex');
line([0,100], [0, 0], "Color", "red");
subplot(2,2,2);
plot(1:100, u_hat);
title('$$\hat{u}$$', 'Interpreter','Latex');
line([0,100], [0, 0], "Color", "red");
subplot(2,2,[3,4]);
plot(X, y, "o", X, y_hat, "-")
title('$$y = X\hat{\beta}$$ vs. $$y = X\beta + u$$', 'Interpreter', 'Latex');

%% 2.1
% nehme gangz willkuerlich an, beta_0 = 3
X = [ones(100, 1), randn(100, 1)];
u = randn(100, 1);
beta = [3; 5];
y = X * beta + u;
beta_hat = inv(X'*X) * X'* y;
y_hat = X * beta_hat;
u_hat = y - y_hat;

% plots
subplot(2,2,1);
plot(1:100, u);
title("$$u$$", 'Interpreter','Latex');
line([0,100], [0, 0], "Color", "red");
subplot(2,2,2);
plot(1:100, u_hat);
title('$$\hat{u}$$', 'Interpreter','Latex');
line([0,100], [0, 0], "Color", "red");
subplot(2,2,[3,4]);
plot(X(:,2), y, "o", X(:,2), y_hat, "-")
title('$$y = X\hat{\beta}$$ vs. $$y = X\beta + u$$', 'Interpreter', 'Latex');

%% 2.1-2.2
[beta_hat, beta_se] = OLS(X, y)

%% 3.
% aquire data (monthly freq)
req = fred;
mnemo = 'CPIAUCSL';
startdate = '01/01/1955';
enddate = '10/01/2020';
raw = fetch(req, mnemo, startdate, enddate);
close(req);

% transform data, pi = annulized inflation in cpi
CPI = raw.Data(:,2);
pi = diff(log(CPI))*100*12;
pi_lag = pi(1:(end-1));
pi = pi(2:end);
Tob = length(pi);
%%
foo = [ones(Tob, 1), pi_lag];
foo_ols = OLS(foo, pi);
mu_hat = foo_ols(1);
b_hat = foo_ols(2);
pi_lr = mu_hat / (1 - b_hat);


%% 4.
j = 50;
pi_lr_rolling = rolling_longrun_AR1(foo, pi, j);
plot(pi_lr_rolling,'LineWidth',2)
title(sprintf('Estimated long-run inflation rate using a rolling window of %i periods', j));
line([0, length(pi_lr_rolling)], [pi_lr, pi_lr], 'Color', 'r', 'Linewidth', 1.5)

%% 5.
Tob = length(pi_lr_rolling)
plot(1:Tob, pi(1:Tob))
line([0, Tob], [pi_lr, pi_lr], 'Color', 'r', 'Linewidth', 1.5);
line([0, length(pi_lr_rolling)], [pi_lr, pi_lr], 'Color', 'r', 'Linewidth', 1.5);
title('Annulized inflation in CPI and estimated static long-run rate')

%%
figure; 
plot(pi(1:size(pi_lr_rolling,1)), 'b', 'LineWidth', 1.5); ylim([-10, 25]); hold on; 
plot(pi_lr_rolling, 'r',  'LineWidth',1.5); 
yline(pi_lr, 'g', 'LineWidth', 1.5);
legend('Actual Inflation Rate', 'LR Inflation Rate (Rolling Window)','Static LR Inflation Rate '); 
title('E2-3/5');
