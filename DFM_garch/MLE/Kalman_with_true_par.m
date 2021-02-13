%%
%makedata
clc
clear
load('data')
%plot(data);
%axis tight;
%% True parameters
% Table 2 Berger&Pozzi(2013)
N = 5;
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

% initialization of Kalman Filter
P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
[PU,PS,PV] = svd(P_0);
rng(37073);
omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

warning('off');
[omega, P, cond_var, LogLike] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,1);
%% sates
load('R_i.mat')
load('R_w.mat')
figure(1);
foo = 1;
subplot(2,3,foo);
plot([omega(:,N+1), R_w], 'Linewidth', 1);
legend('estimate', 'true');title('common factor'); axis tight;
for pn = 1:N
    foo = foo + 1;
    subplot(2,3,foo);
    plot([omega(:,pn), R_i(:,pn)], 'Linewidth', 1);
    legend('estimate', 'true');title("ideosyncratic factor of country " + pn + " "); axis tight;
end

% conditional variances
load('h_i.mat')
load('h_w.mat')
figure(2);
foo = 1;
subplot(2,3,foo);
plot([cond_var(:,N+1), h_w], 'Linewidth', 1);
legend('estimate', 'true');title('conditional variances of common factor'); axis tight;
for pn = 1:N
    foo = foo + 1;
    subplot(2,3,foo);
    plot([cond_var(:,pn), h_i(:,pn)], 'Linewidth', 1);
    legend('estimate', 'true');title("conditional variances of ideosyncratic factor of country " + pn + " "); axis tight;
end


%% does P converges? 
figure(3);
foo = 1;
for pp = 1:12
    subplot(4,3,foo);
    plot(squeeze(P(pp,pp,:)));
    legend('estimate', 'true');title('P'); axis tight;
    foo = foo + 1;
end
