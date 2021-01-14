clc
clear
rng(37073);

Tob     = 200;
N       = 15;   % panel dimension
r       = 1;    % number of common factors
p_f     = 1;    % lag order of common factors
p_e     = 1;    % lag order of idiosyncratic components

Phi_f   = 0.99;
sigma_f = 1.2;
Phi_e   = diag(rand(N,1)*1.8-0.9);
sigma_e = diag(rand(N,1)*0.6+0.6);
Loading = randn(N,r);
Loading = Loading * inv(chol(Loading'*Loading))'; % restriction: Loading'*Loading = eye(r)

for i = 1:r;
    if Loading(i,i) < 0;
        Loading(:,i) = Loading(:,i) * -1;
    end
end

%Loading = ones(N,1);
burn    = 3000;

[y, Fact, eps] = DFM_simu_basic(Tob, r, p_f, p_e, Phi_f, Phi_e, sigma_f, sigma_e, Loading, burn);
save 'y_1.mat',y;

figure1 = figure;
plot(y); hold on;
plot(Fact, 'Linewidth', 4);
title('Observations and Common Factor')
hold off;
saveas(figure1,'SS_DFM1.pdf')
%%
% x = [phi_f, phi_e_1 , ... , phi_e_N, sigma_f, sigma_e_1, ..., sigma_e_N, loading_1, ..., loading_N]
x0 = rand(1, 2*(N+r)+N*r);
options = optimoptions(@fminunc,'MaxIterations',50000);
xopt = fminunc(@DFM_obj1, x0, options);

% display
'True vs. Estimated Factor Loading'
Loading
Load  = reshape(xopt((2*(N+r)+1):end), [N,r]);
Load_est = Load * inv(chol(Load'*Load))'

'True vs. Estimated AR Coeff of Common Factors'
Phi_f
Phi_f_est = diag(tanh(xopt(1:r)))

'True vs. Estimated AR Coeff of Idiosyncratic Components'
Phi_e
Phi_e_est = diag(tanh(xopt((r+1):(N+r))))

'True vs. Estimated Covmat of Common Factors'
sigma_f
Sig_f_est = diag(exp(xopt((N+r+1):(N+r+r))))

'True vs. Estimated Covmat of Idiosyncratic Components'
sigma_e
Sig_e_est = diag(exp(xopt((N+r*2+1):(N+r*2+N))))

% estimtated states
    % state space model representation:
    A   = 0;
    X   = zeros(Tob, 1);
    H   = [Load_est, eye(N)];
    Mu  = zeros((r+N),1);
    F   = blkdiag(Phi_f_est, Phi_e_est);
    Sigma_obs = 1e-10;
    Sigma_trans = blkdiag(Sig_f_est, Sig_e_est);
    
      % initialization
    alpha_0 = [zeros(1, r), zeros(1, N)];
    P_0 = blkdiag(Sig_f_est/(1-Phi_f_est^2),Sig_e_est./(1-Phi_e_est.^2));
    
    % KF
    Fact_est = Kalman_kernel(y, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 0);
    Fact_est =  Fact_est(:,r);
    
    % plot
    figure2 = figure;
    plot([Fact, Fact_est], 'Linewidth', 2);
    legend('true', 'est');
    title('True and Estimated Common Factor');
    saveas(figure2,'SS_DFM2.pdf')

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
    Load  = Load * inv(chol(Load'*Load))';
    
    for i = 1:r;
        if Load(i,i) < 0;
        Load(:,i) = Load(:,i) * -1;
        end
    end
    % state space model representation:
    A   = 0;
    X   = zeros(Tob, 1);
    H   = [Load, eye(N)];
    Mu  = zeros((r+N),1);
    F   = blkdiag(Phi_f, Phi_e);
    Sigma_obs = 1e-10;
    Sigma_trans = blkdiag(Sig_f, Sig_e);
    
    % initialization
    alpha_0 = [zeros(1, r), zeros(1, N)];
    P_0 = blkdiag(Sig_f/(1-Phi_f^2),Sig_e./(1-Phi_e.^2));
    
    % compute log-likelihood
    [~,~,ll] = Kalman_kernel(y, alpha_0', P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, 0);
    
    ll = -ll;

end


