%_________________________%
% Local Level Model: Nile %
%_________________________%

clear
clc
%%
Y = importdata("nile.txt");

[alpha, P, v, F] = local_level_filter(Y, 0, 10^7, 15099, 1469.1);


%%
figure(1);
subplot(2,2,1);
plot(2:100, [Y(2:end), alpha(2:end)], 'LineWidth', 1.5);
title("Data and alpha");
subplot(2,2,2);
plot(2:100, P(2:end));
title("state variance: P");
subplot(2,2,3);
plot(3:100, v(3:end));
title("prediction error: v");
subplot(2,2,4);
plot(3:100, F(3:end));
title("prediction error variance: f");


%%
%addpath('MNZ');
Y = importdata("nile.txt");
Tob = length(Y);
%% Known parameters
A   = 0;
X   = zeros(Tob, 1);
H   = 1;
F   = 1;
Mu  = 0;
Sigma_obs = 15099;
Sigma_trans = 1469.1;
%% init
alpha_0 = 0;
P_0 = 10^7;

[alpha, P, v, f] = Kalman_basic(Y, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 1);

%%
figure(2);
subplot(2,2,1);
plot(2:100, [Y(2:end), alpha(2:end)], 'LineWidth', 1.5);
title("Data and alpha");
subplot(2,2,2);
plot(2:100, squeeze( P(2:end)));
title("state variance: P");
subplot(2,2,3);
plot(3:100, v(3:end));
title("prediction error: v");
subplot(2,2,4);
plot(3:100, squeeze(f(3:end)));
title("prediction error variance: f");

