clc
clear
rng(1234);

Tob     = 200;
N       = 15;    % panel dimension
r       = 1;    % # common factors
p_f     = 1;    % lag order of common factors
p_e     = 1;    % lag order of idiosyncratic components

Phi_f   = 1;
Phi_e   = diag(rand(N,1));
sigma_f = 3;
sigma_e = diag((randn(N,1)).^2+1);
Loading = randn(N,1);
Loading = inv(chol(Loading'*Loading)) * Loading; % restriction: Loading'*Loading = eye(r)
%Loading = ones(N,1);
burn    = 100;

[y, F, eps] = DFM_simu_basic(Tob, r, p_f, p_e, Phi_f, Phi_e, sigma_f, sigma_e, Loading, burn);

plot(y); hold on;
plot(F, 'Linewidth', 2);
hold off;
%%

