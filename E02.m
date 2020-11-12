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
y_hat = beta * X;
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