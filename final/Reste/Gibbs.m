%% 1. simulate data
clc
clear
load('data')

[Tob, N] = size(data);
r = (N+1)*2;
%% 2. Gibbs Sampling
iter    = 8000;
retain  = 2000;
burn    = iter - retain;
T       = Tob-1;

pi_w    = 0.126;
delt_aw = 0.086;
delt_bw = 0.115;
delt_cw = 0.799;

pi_i    = [1e-6, 0.035, -0.075, -0.092, 0.044];
delt_ai = [0.022, 0.001, 0.007, 0.012, 0.026];
delt_bi = [0.133, 0.130, 0.233, 0.089, 0.067];
delt_ci = [0.845, 0.869, 0.760, 0.900, 0.906];

var_eta = [3.34e-4, 3.42e-4, 3.34e-4, 3.41e-4, 3.34e-4];
alpha   = [3.317, 3.430, 5.692, 3.695, 4.609];
beta    = [4.887, 4.501, 3.442, 3.282, 2.766];

% reserve memory
ALPHA   = zeros(N, 1, retain);
BETA    = zeros(N, 1, retain);
PI      = zeros(N+1,1, retain);
DELTA   = zeros(N+1, 3, retain);
VAR_ETA = zeros(N, retain);
FACT    = zeros(T, N+1, retain);
CONDSTD = zeros(T, N+1, retain);

% setting priors
alpha_prior     = 0;
alpha_priors    = 1;
beta_prior      = 0;
beta_priors     = 1;

pii_prior       = 0;
pii_priors      = 0.1;
piw_prior       = 0;
piw_priors      = 0.1;

delta_prior     = 0.5;
delta_priors    = 0.2;

v0      = 4;
delta0  = 0.1;


% 1. sample states conditional on model parameters
% starting values
alpha = randn(N,1);
for i = 1:N;
    if alpha(i) < 0
       alpha(i)=  alpha(i) * -1;
    end
end

beta = randn(N,1);
if beta(1) < 0
   beta = beta * -1;
end

pi_i    = zeros(N,1);
pi_w    = zeros(1);
delt_bw = rand(1)*0.3+0.1;
delt_cw = rand(1)*0.3+0.1;
delt_aw = 1 - delt_bw - delt_cw;
delt_bi = rand(N,1)*0.3+0.1;
delt_ci = rand(N,1)*0.3+0.1;
delt_ai = 1 - delt_bi - delt_ci;
var_eta = ones(N, 1)*1e-4;
%%
for ii = 1:iter
F = diag([pi_i, pi_w, zeros(1,N+1)]);
K = [eye(N+1), eye(N+1)]';
% KF
% initialization of Kalman Filter
P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
[PU,PS,PV] = svd(P_0);
rng(37073);
omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

warning('off');
[omega_11, P_11, cond_var, ~] = kalman_garch(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);

% and recursions
omegatmp = zeros(T, r);
omegas   = zeros(T, r);

% draw 
i         = T;
    
for i = T-1:-1:1
     omegatmp(i+1,:) = omega_11(i+1,:) + randn(1,r) * chol(P_11(:,:,i+1))';
     
     alpha_12  = omega_11(i,:)' + P_11(:,:,i)*F'*inv(F*P_11(:,:,i)*F'+K*diag(cond_var(i,:))*K')*(omegatmp(i+1,1)' - F*omega_11(i,:)');
     P_12      = P_11(:,:,i) - P_11(:,:,i)*F'*inv(F*P_11(:,:,i)*F'+K*diag(cond_var(i,:))*K')*F*P_11(:,:,i);
     omegas(i,:) = alpha_12' + randn(1,r) * chol(P_12)';
end
    
R_I = omegas(:,1:N);
R_W = omegas(:,N+1);
E_I = omegas(:,(N+2):(r-1));
E_W = omegas(:,end);

% 2. pi conditional on states
for nn = 1:N
    
    
% conjugate prior for F, N(0,1)
foo = 0;
F_m = inv(1/phi_fpriors + Alphas(1:T-1)'*Alphas(1:T-1))*(1/phi_fpriors*phi_fprior+ Alphas(1:T-1)'*Alphas(2:T));
F_v = inv(1/phi_fpriors + Alphas(1:T-1)'*Alphas(1:T-1));
while foo == 0
    Phi_f = F_m + sqrt(F_v) * randn(1);
    if abs(Phi_f) < 1
        foo = 1;
    end
end

% 3. sample H (Loading, Phi_e, sigma_e) conditional on states
for n = 1:N
   % sample loading and sigma_e
   TT  = T-1;
   Xn  = Alphas(2:end) - Phi_e(n,n) * Alphas(1:TT);
   Yn  = Y(2:end, n);
   
   % starting values
   lambda_n0   = inv(Xn'*Xn)*Xn'*Yn;
   sigma_en    = (Yn - Xn*lambda_n0)'*(Yn - Xn*lambda_n0)/(TT - 1);
  
   % draw lambda
   lambda_n_m   = inv(inv(loading_priors(n)) + (1/sigma_en) * Xn'*Xn) * (inv(loading_priors(n)) * loading_prior(n) + (1/sigma_en) * Xn'*Yn);
   lambda_n_v   = inv(inv(loading_priors(n)) + (1/sigma_en) * Xn'*Xn);
   Lambda_n     = lambda_n_m + sqrt(lambda_n_v) * randn(1);
   
   % draw sigma_e
   residual_n   = Yn - Xn*Lambda_n;
   v1           = v0(n) + TT;
   delta1       = delta0(n) + residual_n'*residual_n;
   faa          = randn(v1, 1);
   sigma_en     = delta1 / (faa' * faa);
   
   % get the original model residuals
   e_n      = y(2:end,n) - Lambda_n * Alphas;
   e_n_lag  = e_n(1:TT);
   e_n      = e_n(2:end);
   
   % sample phi_e
   phi_e_n_m = inv(inv(phi_epriors(n)) + (1/sigma_en) * e_n_lag'*e_n_lag) * (inv(phi_epriors(n)) * phi_eprior(n) + (1/sigma_en) * e_n_lag'*e_n);
   phi_e_n_v = inv(inv(phi_epriors(n)) + (1/sigma_en) * e_n_lag'*e_n_lag);
   
   foo = 0;
   while foo == 0
       Phi_e_n = phi_e_n_m + sqrt(phi_e_n_v) * randn(1);
       if abs(Phi_e_n) < 1
           foo = 1;
       end
   end
   
   Loading(n)   = Lambda_n;
   Sigma_e(n,n) = sigma_en;
   Phi_e(n,n)   = Phi_e_n;
   
end

Loading = Loading * inv(chol(Loading'*Loading))'; 
for i = 1:r;
    if Loading(i,i) < 0;
        Loading(:,i) = Loading(:,i) * -1;
    end
end

% write outcomes
    if ii > burn
        LOADING(:,:,ii-burn) = Loading;
        FACT(:,:,ii-burn)    = Alphas;
        PHI_F(ii-burn)       = Phi_f;
        PHI_E(:,:,ii-burn)   = Phi_e;
        SIG_E(:,:,ii-burn)   = Sigma_e;
    end

    if mod(ii, 500) == 0
        fprintf('Iter: %g \r', ii)
    end
        
end

%% 2. print and plot