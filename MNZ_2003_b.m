%% Unknwon parameters
clc
clear

% x = [mu, phi_1*, phi_2*, log(sigma_eta), log(sigma_epsilon)]

x0 = [0, 0, 0, 1, 1];

xotp = fminunc(@MNZ_obj, x0);

%% Estimation results
 mu              = xotp(1);
 phi_1           = xotp(2);
 phi_2           = xotp(3);
 sigma_eta       = exp(xotp(4));
 sigma_epsilon   = exp(xotp(5));
 z_1     = phi_1 / (1 + abs(phi_1));
 z_2     = phi_2 / (1 + abs(phi_2));
 phi_1   = z_1 + z_2;
 phi_2   = -1 * z_1 * z_2;

%[0.8119, 1.5305, -0.6097, 0.6893, 0.6199];

% load data again
 load('data.mat');
 Tob = length(data);
 data = data * 100;

% estimated parameters
A               = 0;
X               = zeros(Tob, 1);
H               = [1, 1, 0];
Sigma_obs       = 0;
Mu              = [mu; 0; 0];
F               = [1, 0, 0; 0, phi_1, phi_2; 0, 1, 0];
Sigma_trans     = [sigma_eta, 0, 0; 0, sigma_epsilon, 0; 0, 0, 0];

% initialization
ar2_cor_l1  = phi_1 / (1 - phi_2);
ar2_cor_l2  = phi_1 * ar2_cor_l1 + phi_2;
ar2_var     = sigma_epsilon / (1 - phi_1 * ar2_cor_l1 - phi_2 * ar2_cor_l2);
alpha_0     = [data(1), 0, 0];
P_0         = [10^4, 0, 0; 0, ar2_var, ar2_cor_l1*ar2_var; 0, ar2_cor_l1*ar2_var, ar2_var];

% Filtering
alpha = Kalman_kernel(data, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 0);

subplot(1,2,1);
plot(1:Tob, alpha(:,2), 'LineWidth', 1.5);
xlim([0,Tob]);
line([1,Tob],[0,0]);
title("Estimated Cycle");

subplot(1,2,2);
plot(1:Tob, alpha(:,1), 'LineWidth', 1.5);
xlim([0,Tob]);
title("Estimated Trend");

% display
'Estimated intercept of the transition equation'
Mu

'Estimated state transition matrix'
F

'Estimated covariance matrix of the transition equation'
Sigma_trans

%% Zielfunktion
% x = [mu, phi_1*, phi_2*, log(sigma_eta), log(sigma_epsilon)]
function ll = MNZ_obj(x)

    % load data in namespace
    load('data.mat');
    Tob = length(data);
    data = data * 100;

    % These are restricted by construction!
    A   = 0;
    X   = zeros(Tob, 1);
    H   = [1, 1, 0];
    Sigma_obs = 0;
    
    % define unknowns
    mu              = x(1);
    phi_1           = x(2);
    phi_2           = x(3);
    sigma_eta       = exp(x(4));
    sigma_epsilon   = exp(x(5));
    
    % restrictions for stationary AR(2)
    z_1     = phi_1 / (1 + abs(phi_1));
    z_2     = phi_2 / (1 + abs(phi_2));
    phi_1   = z_1 + z_2;
    phi_2   = -1 * z_1 * z_2;
    
    Mu              = [mu; 0; 0];
    F               = [1, 0, 0; 0, phi_1, phi_2; 0, 1, 0];
    Sigma_trans     = [sigma_eta, 0, 0; 0, sigma_epsilon, 0; 0, 0, 0];

    % initialization
    ar2_cor_l1 = phi_1 / (1 - phi_2);
    ar2_cor_l2 = phi_1 * ar2_cor_l1 + phi_2;
    ar2_var = sigma_epsilon / (1 - phi_1 * ar2_cor_l1 - phi_2 * ar2_cor_l2);
    alpha_0 = [data(1), 0, 0];
    P_0 = [10^4, 0, 0; 0, ar2_var, ar2_cor_l1*ar2_var; 0, ar2_cor_l1*ar2_var, ar2_var];
    
    % compute log-likelihood
    [~,~,ll] = Kalman_kernel(data, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 0);
    
    ll = -ll;
end


    