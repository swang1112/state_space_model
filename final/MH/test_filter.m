clear
load('data.mat')
load('mode.mat')
[alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(mode);


N = 5;

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
