clc
clear
rng(37077);
addpath('funs');

Tob     = 240;
N       = 4;   % panel dimension
r       = 1;    % number of common factors
p_f     = 1;    % lag order of common factors
p_e     = 1;    % lag order of idiosyncratic components

Phi_f_true   = 0.99;
sigma_f_true = 1;
Phi_e_true   = diag([-0.3, 0.2, -0.35, 0.4]);
sigma_e_true = diag([0.3, 0.25, 0.2, 0.15]);
Loading_true = [0.3; 0.4; 0.5; 0.6];
Loading_true = Loading_true * inv(chol(Loading_true'*Loading_true))'; % restriction: Loading'*Loading = eye(r)

for i = 1:r;
    if Loading_true(i,i) < 0;
        Loading_true(:,i) = Loading_true(:,i) * -1;
    end
end

[y, Fact, eps] = DFM_simu_basic(Tob, r, p_f, p_e, Phi_f_true, Phi_e_true, sigma_f_true, sigma_e_true, Loading_true, 3000);

figure0 = figure;
plot(y); hold on;
plot(Fact, 'Linewidth', 4);
title('Observations and Common Factor')
hold off;
saveas(figure0,'Gibbs_DFM0.pdf')

%% Gibbs Sampling
iter    = 8000;
retain  = 2000;
burn    = iter - retain;
T       = Tob-1;

% reserve memory
LOADING = zeros(N,r,retain);
PHI_F   = zeros(1, retain);
PHI_E   = zeros(N, N, retain);
SIG_E   = zeros(N, N, retain);
FACT    = zeros(T, r, retain);


% setting priors
loading_prior   = zeros(N,1);
loading_priors  = ones(N,1);

phi_eprior  = zeros(N,1);
phi_epriors = 0.1*ones(N,1);

v0      = ones(N,1);
delta0  = 0.1*ones(N,1);

phi_fprior  = 0;
phi_fpriors = 0.1;

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
H   = [Loading, -Phi_e*Loading];
F   = [Phi_f, 0; 1, 0];
% transform the observation equations by pre multipyling (eye(N) - Phi_e L): free measurement errors fromautocorrelation 
Y   = y(2:end,:) - y(1:T,:)*Phi_e;
Q   = [1, 0;0, 0];

% initialization of KF
alpha_0 = [0; 0];
P_0 = 1/(1-Phi_f^2)*eye(2);

% KF
%warning('off');
[alpha_11, P_11] = Kalman_kernel(Y, alpha_0, P_0, 0, zeros(T, 1), H, F, 0, Sigma_e, Q, 0);

% and recursions
Alphatmp = zeros(T, 2);
Alphas   = zeros(T, 2);

% draw 
i         = T;
    
for i = T-1:-1:1
     Alphatmp(i+1,:) = alpha_11(i+1,:) + randn(1,2) * chol(P_11(:,:,i+1))';
     
     alpha_12  = alpha_11(i,:)' + P_11(:,:,i)*F(1,:)'*inv(F(1,:)*P_11(:,:,i)*F(1,:)'+Q(1,1))*(Alphatmp(i+1,1)' - F(1,:)*alpha_11(i,:)');
     P_12      = P_11(:,:,i) - P_11(:,:,i)*F(1,:)'*inv(F(1,:)*P_11(:,:,i)*F(1,:)'+1)*F(1,:)*P_11(:,:,i);
     Alphas(i,:) = alpha_12' + randn(1,2) * chol(P_12)';
end
    
Alphas = Alphas(:,1);

% 2. sample F (Phi_f) conditional on states
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


Geweke_test_Loading = zeros(N,2);
figure1 = figure(1);
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
    [z0, pval0] = geweke(squeeze(LOADING(n,:,:)), 0.1, 0.5);
    Geweke_test_Loading(n,:)  = [z0, pval0]; 
end
saveas(figure1,'Gibbs_DFM1.pdf')

Geweke_test_Phi_e = zeros(N,2);
figure2 = figure(2);
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
    [z0, pval0] = geweke(squeeze(PHI_E(n,n,:)), 0.1, 0.5);
    Geweke_test_Phi_e(n,:) = [z0, pval0]; 
end
saveas(figure2,'Gibbs_DFM2.pdf')

Geweke_test_sigma_e = zeros(N,2);
figure3 = figure(3);
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
    [z0, pval0] = geweke(squeeze(SIG_E(n,n,:)), 0.1, 0.5);
    Geweke_test_sigma_e(n,:) = [z0, pval0];
end
saveas(figure3,'Gibbs_DFM3.pdf')

figure4 = figure(4);
subplot(2,2,1);
hist(squeeze(PHI_F), 50);
title("Phi_f (true: " + Phi_f_true + " )");
subplot(2,2,2);
plot(squeeze(PHI_F));
subplot(2,2,3);
autocorr(squeeze(PHI_F));
subplot(2,2,4);
plot([Fact(2:end), mean(FACT,3)], 'Linewidth', 2);
legend('True','Estimated');
title('True and Estimated Common Factor');
[z0, pval0] = geweke(squeeze(PHI_F), 0.1, 0.5);
Geweke_test_PHI_F = [z0, pval0];
saveas(figure4,'Gibbs_DFM4.pdf')