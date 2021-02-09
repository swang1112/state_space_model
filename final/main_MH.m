%% MLE as starting value
clear
clc
addpath('fun')
load('xopt.mat')
load('data.mat')
load('hessian.mat')
%%
par_old = xopt';

lr = 0.01;         % learning rate
S  = eye(33) * lr; % update covaraiance

% setting priors
% i) prior for [alpha'; beta']: gaussian
loading0 = 0.1;
loading0s= 0.5;
% ii) prior for [pi_i'; pi_w]: gaussian
ar0 = 0;
ar0s= 0.3;
% iii) prior for [delt_bi'; delt_ci'; delt_bw; delt_cw]: IG
garch_v0 = 1;
garch_d0 = 1;
% iv) prior for var_eta: IG
eta_v0 = 1;
eta_d0 = 0.01;

% old poseterior
[alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par_old');
% initialization of Kalman Filter
P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
[PU,PS,PV] = svd(P_0);
rng(37073); 
omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

[~, ~, ~, ll] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);
    
foo = 0;
for j = 1:5
    foo = foo + foo + lgam(garch_v0, garch_d0, 1./delt_bi(j)) + lgam(garch_v0, garch_d0, 1./delt_ci(j)) + lgam(eta_v0, eta_d0, 1./var_eta(j));
end
post_old = ll + foo + sum(log(mvnpdf([alpha'; beta'], loading0, loading0s))) + sum(log(mvnpdf([pi_i'; pi_w'], ar0, ar0s))) + lgam(garch_v0, garch_d0, 1./delt_bw) +  lgam( garch_v0, garch_d0, 1./delt_cw);

iter = 4000;
burn = 3000;
nacc = 0; % # total draws
out = zeros(iter-burn,33); % reserve memory
%%
for ii = 1:iter
    % 1. propose new par
    par_new = par_old + (randn(1,33)*S)';
    
    % 2. evaluate posterior at new draw
    [alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par_new');
    % check parameter restrictions
    if abs(pi_i) > 1 | abs(pi_w) >1 | delt_bw < 0 | delt_bw > 1 | delt_cw < 0 | delt_cw > 1 | delt_aw < 0 | delt_bi < 0 | delt_bi > 1 | delt_ci < 0 | delt_ci > 1 | delt_ai < 0 | var_eta < 0 
        acc_prob = 0;
    else
        % initialization of Kalman Filter
        P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
        [PU,PS,PV] = svd(P_0);
        rng(37073); 
        omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

        warning('off')
        [~, ~, ~, ll] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);
    
        foo = 0;
        for j = 1:5
            foo = foo + lgam(garch_v0, garch_d0, 1./delt_bi(j)) + lgam(garch_v0, garch_d0, 1./delt_ci(j)) + lgam(eta_v0, eta_d0, 1./var_eta(j));
        end
        post_new = ll + foo + sum(log(mvnpdf([alpha'; beta'], loading0, loading0s))) + sum(log(mvnpdf([pi_i'; pi_w'], ar0, ar0s))) + lgam(garch_v0, garch_d0, 1./delt_bw) +  lgam( garch_v0, garch_d0, 1./delt_cw);
    
        % 3. decide accept/discard the draw 
        acc_prob = min([exp(post_new - post_old); 1]);   
    end
    u        = rand(1);
    if u < acc_prob
        par_old     = par_new;  
        post_old    = post_new;
        nacc        = nacc+1;  
    end
    
    acc_rate = nacc/ii;
    % lerning rate scheduling
    if ii > 1000 && ii < 2000  
      if acc_rate > 0.4
          S = S * 1.0000001;
      elseif acc_rate<0.2
          S = S * 0.99;
      end
    end
      
    % store results
    if ii > burn
        out(ii-burn,:) = par_old';
    end

    if mod(ii, 500) == 0
        fprintf('Iter: %g \r', ii)
    end
    
end
disp(acc_rate)
%% print and plot

figure1 = figure;
foo = 1;
for n = 1:5
    subplot(5,3,foo);
    hist(out(:,n+3), 50);
    %title("Loadings (true: " + Loading_true(n) + " )");
    foo = foo + 1;
    subplot(5,3,foo);
    plot(out(:,n+3));
    foo = foo + 1;
    subplot(5,3,foo);
    autocorr(out(:,n+3));
    foo = foo + 1;
    %[z0, pval0] = geweke(squeeze(LOADING(n,:,:)), 0.1, 0.5);
    %Geweke_test_Loading(n,:)  = [z0, pval0]; 
end
%saveas(figure1,'Gibbs_DFM1.pdf')

%%

Geweke_test_Phi_e = zeros(N,2);
figure2 = figure;
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
figure3 = figure;
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

figure4 = figure;
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

