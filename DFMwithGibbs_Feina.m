%% DFM with Gibbs Sampling 
% common factor: G_t
% series-specific factor: eps_jt

% y_jt = phi_j * G_t + eps_jt
% G_t = rho * G_t-1 + v_t, v_t ~ N(0, 1)                   
% eps_jt = gamma_j * eps_jt + eta_jt, eta_jt ~ N(0, sigma_eta_j);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc; 
addpath('Functions'); 

%% Data simulation
rng(100); 
T = 200; 
N = 5; 
 
% common factor
rho = rand(1, 1); % AR coefficient 
% identification restriction
phi = [rand(1, 1); randn(N-1, 1)]; % factor loadings
sigma_v = 1;  % variance of error term 

% series specific factors
gamma = rand(N, 1)*0.95; % AR coefficient 
sigma_eta = rand(N, 1); % variance of error term

theta_true = [phi; rho; gamma; sigma_eta]; 

[y, G, eps, eta] = DFM_simul(T, N, rho, phi, gamma, sigma_eta, sigma_v); 
 


%% Gibbs Sampling 
nd = 5000; % number of retained draws
burn = 4000; % burn-in


% Priors
a0_rho = 0.5;
A0_rho = 0.05^2; 
a0_phi = ones(N, 1) * 0.5; % [rand(1, 1); randn(N-1, 1)]; 
A0_phi = [0.5^2 0 0 0 0 ; 
          0 1^2 0 0 0 ; 
          0 0 0.25^2 0 0 ; 
          0 0 0 1^2 0 ; 
          0 0 0 0 0.25^2]; % eye(N) * (0.5^2); 
a0_gamma = ones(N, 1) * 0.5;  % rand(N, 1)*0.95; 
A0_gamma = [0.5^2 0 0 0 0 ; 
            0 0.1^2 0 0 0 ; 
            0 0 0.25^2 0 0 ; 
            0 0 0 0.5^2 0 ; 
            0 0 0 0 0.1^2];   % eye(N) * (1^2); 
bel_sigma_eps = ones(N, 1) * 0.5; 
str_sigma_eps = [0.005; 0.005; 0.005; 0.05; 0.005];    % ones(N, 1) * 0.005; 
d0 = T*str_sigma_eps; 
D0 = diag(d0 .* bel_sigma_eps);

a1_prior = [a0_phi; a0_rho; a0_gamma]; 
b1_prior = [diag(A0_phi); diag(A0_rho); diag(A0_gamma)]; 
CI90_N = CI90(a1_prior, b1_prior, T, nd, 1); 

CI90_IG = CI90(bel_sigma_eps, str_sigma_eps, T, nd, 2); 

prior_CI90 = [CI90_N; CI90_IG]; 

% starting values
theta = [randn(N, 1); rand(1,1); randn(N,1); rand(N,1)]; 

% save draws
store_parameters = zeros(3*N+1, nd); 

for i = 1:burn+nd
    
    if (mod(i, 100) ==0)
       disp([num2str( i) ' loops... ']);
    end

%% Step 1: Sample G_t
% Construct state space matrices
phi = theta(1:N, : ); 
rho = theta(N+1, :); 
gamma = theta(N+2:2*N+1,:); 
sigma_eta = theta(2*N+2:3*N+1,:);  
sigma_v = 1;

H = [phi -phi.*gamma];  
F = [rho 0; 0 1]; 
Q = [1 0; 0 0]; 
R = diag(sigma_eta); 

y_data = y'; 
x_data = []; 
mu = zeros(2,1); 
A = []; 

beta_00 = zeros(2, 1); 
P_00 = eye(2) * 1/(1-rho^2); % (both States have the same unconditional distribution)

[beta_tt, P_tt, ML] = kalman(beta_00, P_00, A, H, R, Q, F, mu, y_data, x_data, 2); 
beta_tt = squeeze(beta_tt);
P_tt = squeeze(P_tt); 

%%%%%%%%%%%%%%%%%%%%%
% Since Q is singular
%%%%%%%%%%%%%%%%%%%%%
beta_tT = zeros(2, T);
P_tT = zeros(2, 2, T); 

beta_post = zeros(2, T); 
beta_post(:, T)  = beta_tt(:, end) + chol(squeeze(P_tt(:,:, end))) * randn(2,1) ; % Kim and Nelson P195 Eq8.9

J = 1; 
F_star = F(1:J, :); 
Q_star = Q(1:J, 1:J); 


for t = T-1:-1:1
    
beta_tT(:, t) = beta_tt(:, t) + P_tt(:,:, t) * F_star' * inv(F_star * P_tt(:,:, t) * F_star' + Q_star) * (beta_post(1:J, t+1) - F_star * beta_tt(:, t)); 
P_tT(:,:,t) = P_tt(:,:, t) - P_tt(:,:, t) * F_star' * inv(F_star * P_tt(:,:, t) * F_star' + Q_star) * F_star * P_tt(:,:, t); 
beta_post(:, t) = beta_tT(:, t) + chol(squeeze(P_tT(:,:,t))) * randn(2,1); 

end

G_post = beta_post(1:J, :)';


%% Step 2: Sampling rho conditional on G_t
y1 = G_post(2:end); 
x1 = G_post(1:end-1); 
co1 = 0; 
rho1 = N_post(x1, y1, 1, a0_rho, A0_rho); 
    if rho1 < 1;
        rho_post = rho1; 
    else co1 = co1 + 1; 
    end
    if co1 == 50; 
        rho_post = rho; 
    end
    
% Step 3.1: Sampling phi_j
y2 = ones(T-1, N); 
x2 = ones(T-1, N);
phi_post = ones(N, 1); 

for j = 1:N
    y2(:, j) = y(2:end, j) - gamma(j) * y(1:end-1, j); 
    x2(:, j) = G(2:end, :) - gamma(j) * G(1:end-1, :); 
    phi_post(j) = N_post(x2(:, j), y2(:, j), sigma_eta(j), a0_phi(j, :), A0_phi(j,j)); 
end

% Step 3.2 Sampling gamma_j
eps_post = ones(T, N); 
gamma_post = ones(N, 1); 
y3 = ones(T-1, N); 
x3 = ones(T-1, N);
co2 = 0; 
for j = 1:N
    eps_post(:, j) = y(:, j) - phi_post(j) * G; 
    y3(:, j) = eps_post(2:end, j); 
    x3(:, j) = eps_post(1:end-1, j);
    gamma1 = N_post(x3(:, j), y3(:, j), sigma_eta(j), a0_gamma(j, :), A0_gamma(j,j)); 
    if gamma1 < 1;
        gamma_post(j) = gamma1; 
    else co2 = co2 + 1; 
    end
    if co2 == 50; 
        gamma_post(j) = gamma(j); 
    end
end

% Step 3.3 Sampling sigma_eta
sigma_eta_post = ones(N, 1); 
res = ones(T-1, N); 
for j = 1:N
    res(:, j) = eps_post(2:end, j) - gamma_post(j) * eps_post(1:end-1, j);
    sigma_eta_post(j) = IG_post(d0(j), D0(j,j), res(:, j)); 
end


theta_post = [phi_post; rho_post; gamma_post; sigma_eta_post]; 

theta = theta_post; 

if i>burn  
    store_parameters(1:N, i-burn) = phi_post; 
    store_parameters(N+1, i-burn) = rho_post; 
    store_parameters(N+2:2*N+1, i-burn) = gamma_post; 
    store_parameters(2*N+2:3*N+1, i-burn) = sigma_eta_post;  
end



end

%%
figure; plot(G, 'LineWidth', 1.5); hold on; plot(G_post, 'r', 'LineWidth', 1.5); legend('G', 'G post'); title('Common Factor'); 

%%
estimated_posterior_parameters    =  prctile(store_parameters, 50, 2); 

low5 = prctile(store_parameters, 5, 2); 
upper95 = prctile(store_parameters, 95, 2); 
Posteriors_CI90 = [low5, upper95]; 

names1 = strings(N, 1); names2 = strings(N, 1); names3 = strings(N, 1); names4 = strings(N, 1); 
for j = 1:N
    names1(j) = "phi" + num2str(j);
    names2(j) = "gamma" + num2str(j);
    names3(j) = "sigma eta" + num2str(j);
end
names = [names1; "rho"; names2; names3]; 

table(names, theta_true, prior_CI90, estimated_posterior_parameters, Posteriors_CI90)
 
histo(store_parameters',names,theta_true)


%% Check the convergence of Gibbs Sampler

% 1) Integrated Autocorrelation Time
tau = zeros(3*N+1, 1);
EffectiveDraws = zeros(3*N+1, 1); 
for j = 1:3*N+1
    tau(j) = IF_GE(store_parameters(j,:)');
    EffectiveDraws(j) = nd/tau(j);  
end
display 'Integrated Autocorrelation Time (tau)' ; 
table(names, tau, EffectiveDraws)

% 2) Trace Plots
trace(store_parameters', names, nd)

% 3) Geweke's Test
Geweke_Z = zeros(3*N+1, 1); 
Convergence = zeros(3*N+1, 1);

for j = 1:3*N+1
    
    Geweke_Z(j) = geweke_diagnostic(store_parameters(j, :)'); 
    
    if abs(Geweke_Z(j)) < 1.96; % 5%
        Convergence(j) = 1; % converge
    else
       Convergence(j) = 0; % not converge
    end
end
table(names, Geweke_Z, Convergence)


% 4) Plot Autocorrelation 
% auto(store_parameters', names)

% 5) Recursive Means
% Rmeans(store_parameters', nd, burn, names)









