%% Known parameters
clc
clear
load('data.mat');
Tob = length(data);
data = data * 100;

A   = 0;
X   = zeros(Tob, 1);
H   = [1, 1, 0];
Mu  = [0.8119; 0; 0];
Sigma_obs = 0;
Sigma_trans = [0.6893, 0, 0; 0, 0.6199, 0; 0, 0, 0];

phi_1 = 1.5305;
phi_2 = -0.6097;
F   = [1, 0, 0; 0, phi_1, phi_2; 0, 1, 0];

ar2_cor_l1 = phi_1 / (1 - phi_2);
ar2_cor_l2 = phi_1 * ar2_cor_l1 + phi_2;
ar2_var = 0.6199 / (1 - phi_1 * ar2_cor_l1 - phi_2 * ar2_cor_l2);


alpha_0 = [data(1), 0, 0];
P_0 = [10^4, 0, 0; 0, ar2_var, ar2_cor_l1*ar2_var; 0, ar2_cor_l1*ar2_var, ar2_var];
alpha = Kalman_kernel(data, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 0);

subplot(1,2,1);
plot(1:Tob, alpha(:,2), 'LineWidth', 1.5);
xlim([0,Tob]);
line([1,Tob],[0,0]);
title("Cycle");

subplot(1,2,2);
plot(1:Tob, alpha(:,1), 'LineWidth', 1.5);
xlim([0,Tob]);
title("Trend");





