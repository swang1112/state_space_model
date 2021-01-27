%% 1. simulate data
makedata
%% 2. MLE parameter estimation
clc
clear

MLE

%% 3. estimation results
N = 5;
est_pi_w    = xopt(1);
est_delt_bw = xopt(2);
est_delt_cw = xopt(3);
est_delt_aw = 1 - est_delt_bw - est_delt_cw;

est_pi_i    = xopt(4:8);
est_delt_bi = xopt(9:13);
est_delt_ci = xopt(14:18);
est_delt_ai = 1 - est_delt_bi - est_delt_ci;

est_var_eta = xopt(19:23);
est_alpha   = xopt(24:28);
est_beta    = xopt(29:33);

% sign restrictions
for nn = 1:N
    if est_alpha(nn) < 0
        est_alpha(nn) = -est_alpha(nn);
    end
end
if est_beta(1) < 0
   est_beta = -est_beta; 
end

% display
% True parameters in Table 2 Berger&Pozzi(2013)
'True vs. Estimated common factor ar coeff'
0.126
est_pi_w

'True vs. Estimated common factor loading'
[4.887, 4.501, 3.442, 3.282, 2.766]
est_beta

'True vs. Estimated common factor GARCH parameters'
[0.086, 0.115, 0.799]
[est_delt_aw, est_delt_bw, est_delt_cw]

'True vs. Estimated country-specific ar coeff'
[1e-4, 0.035, -0.075, -0.092, 0.044]
est_pi_i

'True vs. Estimated country-specific loading'
[3.317, 3.430, 5.692, 3.695, 4.609]
est_alpha

'True vs. Estimated country-specific GARCH parameters'
[0.022, 0.001, 0.007, 0.012, 0.026]
est_delt_ai
[0.133, 0.130, 0.233, 0.089, 0.067]
est_delt_bi
[0.845, 0.869, 0.760, 0.900, 0.906]
est_delt_ci

'True vs. Estimated measurement error variances'
[3.34e-4, 3.42e-4, 3.34e-4, 3.41e-4, 3.34e-4]
est_var_eta

%% 4. Latent factors and their conditional variances
load('data')
P_0     = [diag([1./(1-est_pi_i.^2), 1/(1-est_pi_w^2)]), eye(6); eye(6), eye(6)];
[PU,PS,PV] = svd(P_0);
rng(37073); 
omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

[ometa_est,~,h_est] = kalman_garch(data, omega_0, P_0, est_alpha, est_beta, est_pi_i, est_pi_w, est_delt_aw, est_delt_bw, est_delt_cw, est_delt_ai, est_delt_bi, est_delt_ci, est_var_eta);

% sates
load('R_i.mat')
load('R_w.mat')
figure1=figure(1);
foo = 1;
subplot(2,3,foo);
plot([ometa_est(:,N+1), R_w], 'Linewidth', 1);
legend('estimate', 'true');title('common factor'); axis tight;
for pn = 1:N
    foo = foo + 1;
    subplot(2,3,foo);
    plot([ometa_est(:,pn), R_i(:,pn)], 'Linewidth', 1);
    legend('estimate', 'true');title("ideosyncratic factor of country " + pn + " "); axis tight;
end


% conditional variances
load('h_i.mat')
load('h_w.mat')
figure2=figure(2);
foo = 1;
subplot(2,3,foo);
plot([h_est(:,N+1), h_w], 'Linewidth', 1);
legend('estimate', 'true');title('conditional variances of common factor'); axis tight;
for pn = 1:N
    foo = foo + 1;
    subplot(2,3,foo);
    plot([h_est(:,pn), h_i(:,pn)], 'Linewidth', 1);
    legend('estimate', 'true');title("conditional variances of ideosyncratic factor of country " + pn + " "); axis tight;
end

saveas(figure1, 'states.pdf')
saveas(figure2, 'cond_var.pdf')
