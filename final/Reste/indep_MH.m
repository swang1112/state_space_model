%% MLE as starting value
clear
clc
addpath('fun')
load('xopt.mat')
load('data.mat')
%%
par_old = xopt';

% setting priors
% i) prior for [alpha'; beta']: gaussian
loading0 = zeros(10,1)+0.1;
loading0s= eye(10)*0.5;
% ii) prior for [pi_i'; pi_w]: gaussian
ar0 = zeros(6,1);
ar0s= eye(6)*0.3;
% iii) prior for [delt_bi'; delt_ci'; delt_bw; delt_cw]: IG
garch_v0 = 5;
garch_d0 = 0.3;
% iv) prior for var_eta: IG
eta_v0 = 5;
eta_d0 = 0.01;

% old poseterior
post_old = - neg_posterior(par_old',data,loading0,loading0s,ar0,ar0s,garch_v0,garch_d0,eta_v0,eta_d0);

iter = 30;
burn = 20;
nacc = 0; % # total draws
out = zeros(iter-burn,33); % reserve memory

%%
for ii = 1:iter
    % 1. propose new par
    par0 = par_old;
    [par_new, fval] = fminunc(@obj,par0);
    
    % 2. evaluate posterior at new draw
    [alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par_new');
    % check parameter restrictions
    check = sum(abs(pi_i) > 1) + sum(abs(pi_w) >1) + sum(delt_bw < 0) + sum(delt_bw > 1) + sum(delt_cw < 0) + sum(delt_cw > 1) + sum(delt_aw < 0) + sum(delt_bi < 0) + sum(delt_bi > 1) + sum(delt_ci < 0) + sum(delt_ci > 1) + sum(delt_ai < 0) + sum(var_eta < 0 );
    if check == 0
        post_new = -fval;
    
        % 3. decide accept/discard the draw 
        acc_prob = min([exp(post_new - post_old); 1]);   
    else
        acc_prob = 0;
    end
    u        = rand(1,1);
    if u < acc_prob
        par_old     = par_new;  
        post_old    = post_new;
        nacc        = nacc+1;  
    end
    
    acc_rate = nacc/ii;
      
    % store results
    if ii > burn
        out(ii-burn,:) = par_old';
    end

    if mod(ii, 20) == 0
        fprintf('Iter: %g \r', ii)
        disp(nacc)
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

%% posterior
function out = obj(par)
par = par';
load('data.mat')
% setting priors
% i) prior for [alpha'; beta']: gaussian
loading0 = zeros(10,1)+0.1;
loading0s= eye(10)*0.5;
% ii) prior for [pi_i'; pi_w]: gaussian
ar0 = zeros(6,1);
ar0s= eye(6)*0.3;
% iii) prior for [delt_bi'; delt_ci'; delt_bw; delt_cw]: IG
garch_v0 = 5;
garch_d0 = 0.3;
% iv) prior for var_eta: IG
eta_v0 = 5;
eta_d0 = 0.01;

out = neg_posterior(par,data,loading0,loading0s,ar0,ar0s,garch_v0,garch_d0,eta_v0,eta_d0);
end
