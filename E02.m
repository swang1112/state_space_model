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
my_url = 'https://fred.stlouisfed.org/';
req = fred(my_url);
mnemo = 'CPALTT01USM661S';
data = fetch(req, mnemo);
close(req);





