clc
clear
rng(1234);

Tob     = 200;
N       = 5;   % panel dimension
r       = 1;    % number of common factors
p_f     = 1;    % lag order of common factors
p_e     = 1;    % lag order of idiosyncratic components

Phi_f   = 0.99;    % common factors follow random walk;
Phi_e   = diag(rand(N,1)*2-1);
sigma_f = 3;
sigma_e = diag((randn(N,1)).^2+1);
Loading = randn(N,1);
Loading = inv(chol(Loading'*Loading)) * Loading; % restriction: Loading'*Loading = eye(r)
%Loading = ones(N,1);
burn    = 100;

[y, Fact, eps] = DFM_simu_basic(Tob, r, p_f, p_e, Phi_f, Phi_e, sigma_f, sigma_e, Loading, burn);
save 'y_1.mat',y;

plot(y); hold on;
plot(Fact, 'Linewidth', 4);
title('observables and their common factor')
hold off;
%%
% x = [phi_f, phi_e_1 , ... , phi_e_N, sigma_f, sigma_e_1, ..., sigma_e_N, loading_1, ..., loading_N]
x0 = rand(1, 2*(N+r)+N*r);
xopt = fminunc(@DFM_obj1, x0);

% display
'True vs. Estimated Factor Loading'
Loading
Load  = reshape(xopt((2*(N+r)+1):end), [N,r]);
inv(chol(Load'*Load)) * Load

'True vs. Estimated AR Coeff of Common Factors'
Phi_f
diag(tanh(xopt(1:r)))

'True vs. Estimated AR Coeff of Idiosyncratic Components'
Phi_e
diag(tanh(xopt((r+1):(N+r))))

'True vs. Estimated Covmat of Common Factors'
sigma_f
diag(exp(xopt((N+r+1):(N+r+r))))

'True vs. Estimated Covmat of Idiosyncratic Components'
sigma_e
diag(exp(xopt((N+r*2+1):(N+r*2+N))))


%% Zielfunktion
% x = [phi_f, phi_e_1 , ... , phi_e_N, sigma_f, sigma_e_1, ..., sigma_e_N, loading_1, ..., loading_N]

function ll = DFM_obj1(x)

    % load data in namespace
    load('y_1.mat');
    [Tob, N] = size(y);

    % DFM spesifications:
    r       = 1;    % number of common factors
    p_f     = 1;    % lag order of common factors
    p_e     = 1;    % lag order of idiosyncratic components
    
    % DFM parameter matrices:
    Phi_f = diag(tanh(x(1:r)));
    Phi_e = diag(tanh(x((r+1):(N+r))));
    Sig_f = diag(exp(x((N+r+1):(N+r+r))));
    Sig_e = diag(exp(x((N+r*2+1):(N+r*2+N))));
    Load  = reshape(x((2*(N+r)+1):end), [N,r]);
    Load  = inv(chol(Load'*Load)) * Load;
    
    % state space model representation:
    A   = 0;
    X   = zeros(Tob, 1);
    H   = [Load, eye(N)];
    Mu  = zeros((r+N),1);
    F   = blkdiag(Phi_f, Phi_e);
    Sigma_obs = 1e-10;
    Sigma_trans = blkdiag(Sig_f, Sig_e);
    
    % initialization
    alpha_0 = [repmat(y(1), [1,r]), zeros(1, N)];
    P_0 = blkdiag(diag(repmat(10^4, [1,r])),Sig_e./(1-Phi_e.^2));
    
    % compute log-likelihood
    [~,~,ll] = Kalman_kernel(y, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 0);
    
    ll = -ll;

end


