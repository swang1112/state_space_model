clc
clear
rng(37073);
addpath('funs');

Tob     = 240;
N       = 4;   % panel dimension
r       = 1;    % number of common factors
p_f     = 1;    % lag order of common factors
p_e     = 1;    % lag order of idiosyncratic components

Phi_f_true   = 0.99;
sigma_f_true = 1;
Phi_e_true   = diag(rand(N,1)*1.4-0.7);
sigma_e_true = diag(rand(N,1)*0.6+0.1);
Loading_true = randn(N,r);
Loading_true = Loading_true * inv(chol(Loading_true'*Loading_true))'; % restriction: Loading'*Loading = eye(r)

for i = 1:r;
    if Loading_true(i,i) < 0;
        Loading_true(:,i) = Loading_true(:,i) * -1;
    end
end

[y, Fact, eps] = DFM_simu_basic(Tob, r, p_f, p_e, Phi_f_true, Phi_e_true, sigma_f_true, sigma_e_true, Loading_true, 3000);

figure1 = figure;
plot(y); hold on;
plot(Fact, 'Linewidth', 4);
title('Observations and Common Factor')
hold off;


%% Gibbs Sampling
iter    = 5000;
burn    = iter - 1000;
T       = Tob-1;

% reserve memory
LOADING = zeros(N,r,1000);
PHI_F   = zeros(1, 1000);
PHI_E   = zeros(N, N, 1000);
SIG_E   = zeros(N, N, 1000);
FACT    = zeros(T, r, 1000);


% setting priors
loading_prior   = 0; % for all n = 1, .., N
loading_priors  = 1;

phi_fprior  = 0;
phi_fpriors = 1;

phi_eprior  = 0; % for all n = 1, .., N
phi_epriors = 0.2;

v0      = 1; % for all n = 1, .., N
delta0  = 0.1;

% 1. sample states conditional on model parameters
% starting values
loading0 = randn(N,r);
loading0 = loading0 * inv(chol(loading0'*loading0))';

for i = 1:r;
    if loading0(i,i) < 0;
        loading0(:,i) = loading0(:,i) * -1;
    end
end

phi_f0 = 0.5;
phi_e0 = zeros(N);
sigma_e0 = eye(N);

Loading = loading0;
Phi_f   = phi_f0;
Phi_e   = phi_e0;
Sigma_e = sigma_e0;

for ii = 1:iter
% state space model representation:
H   = Loading;
F   = Phi_f;
% transform the observation equations by pre multipyling (eye(N) - Phi_e L): free measurement errors fromautocorrelation 
Y   = y(2:end,:) - y(1:T,:)*Phi_e;
H   = (eye(N) - Phi_e)*H;

% initialization of KF
alpha_0 = 0;
P_0 = 1/(1-Phi_f^2);
    
% KF
%warning('off');
[alpha_11, P_11] = Kalman_kernel(Y, alpha_0, P_0, 0, zeros(T, 1), H, F, 0, Sigma_e, 1, 0);

% and recursions
Alpha01 = zeros(T, 1);
Alphas  = zeros(T, 1);

% draw
for i = T-1:-1:1
     Alpha01(i+1) = alpha_11(i+1) + sqrt(P_11(i+1)) * randn(1);
     alpha_12  = alpha_11(i) + P_11(i)*F'*inv(F*P_11(i)*F'+1)*(Alpha01(i+1) - F*alpha_11(i));
     P_12      = P_11(i) - P_11(i)*F'*inv(F*P_11(i)*F'+1)*F*P_11(i);
     Alphas(i) = alpha_12 + sqrt(P_12) * randn(1);
end
    
% 2. sample F (Phi_f) conditional on states
% conjugate prior for F, N(0,1)
foo = 0;
F_m = inv(1 + Alphas(1:T-1)'*Alphas(1:T-1))*(Alphas(1:T-1)'*Alphas(2:T));
F_v = inv(1 + Alphas(1:T-1)'*Alphas(1:T-1));
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
   lambda_n_m   = inv(inv(loading_priors) + (1/sigma_en) * Xn'*Xn) * (inv(loading_priors) * loading_prior + (1/sigma_en) * Xn'*Yn);
   lambda_n_v   = inv(inv(loading_priors) + (1/sigma_en) * Xn'*Xn);
   Lambda_n     = lambda_n_m + sqrt(lambda_n_v) * randn(1);
   
   % draw sigma_e
   residual_n   = Yn - Xn*Lambda_n;
   v1           = v0 + TT;
   delta1       = delta0 + residual_n'*residual_n;
   faa          = randn(v1, 1);
   sigma_en     = delta1 / (faa' * faa);
   
   % get the original model residuals
   e_n      = y(2:end,n) - Lambda_n * Alphas;
   e_n_lag  = e_n(1:TT);
   e_n      = e_n(2:end);
   
   % sample phi_e
   phi_e_n_m = inv(inv(phi_epriors) + (1/sigma_en) * e_n_lag'*e_n_lag) * (inv(phi_epriors) * phi_eprior + (1/sigma_en) * e_n_lag'*e_n);
   phi_e_n_v = inv(inv(phi_epriors) + (1/sigma_en) * phi_epriors'*phi_epriors);
   
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

end

%% print and plot
Loading_true
mean(LOADING,3)
prctile(LOADING,[5, 95],3)

Phi_f_true
mean(PHI_F)
prctile(PHI_F, [5, 95])

Phi_e_true
mean(PHI_E,3)
prctile(PHI_E,[5, 95],3)

sigma_e_true
mean(SIG_E,3)
prctile(SIG_E,[5, 95],3)

figure(1);
foo = 1;
for n = 1:N
    subplot(N,3,foo);
    hist(squeeze(LOADING(n,:,:)), 50);
    title("Loadings (true: " + Loading_true(n) + " )");
    foo = foo + 1;
    subplot(N,3,foo);
    plot(squeeze(LOADING(n,:,:)));
    foo = foo + 1;
    subplot(N,3,foo);
    autocorr(squeeze(LOADING(n,:,:)));
    foo = foo + 1;
end

figure(2);
foo = 1;
for n = 1:N
    subplot(N,3,foo);
    hist(squeeze(PHI_E(n,n,:)), 50);
    title("phi_e (true: " + Phi_e_true(n,n) + " )");
    foo = foo + 1;
    subplot(N,3,foo);
    plot(squeeze(PHI_E(n,n,:)));
    foo = foo + 1;
    subplot(N,3,foo);
    autocorr(squeeze(PHI_E(n,n,:)));
    foo = foo + 1;
end

figure(3);
foo = 1;
for n = 1:N
    subplot(N,3,foo);
    hist(squeeze(SIG_E(n,n,:)), 50);
    title("sigma_e (true: " + sigma_e_true(n,n) + " )");
    foo = foo + 1;
    subplot(N,3,foo);
    plot(squeeze(SIG_E(n,n,:)));
    foo = foo + 1;
    subplot(N,3,foo);
    autocorr(squeeze(SIG_E(n,n,:)));
    foo = foo + 1;
end

figure(4);
subplot(2,2,1);
hist(squeeze(PHI_F), 50);
title("Phi_f (true: " + Phi_f_true + " )");
subplot(2,2,2);
plot(squeeze(PHI_F));
subplot(2,2,3);
autocorr(squeeze(PHI_F));
subplot(2,2,4);
plot([Fact(2:end), mean(FACT,3)], 'Linewidth', 4);
legend('True','Estimated');
title('True and Estimated Common Factor')
