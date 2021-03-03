%% 1. simulate data
makedata

%% 2. initial estimate (mode of posterior)
init_est

%% 3. MH 
clear
clc
addpath('fun');
load('data.mat');
load('mode.mat');
load('post_val.mat');
load('post_hess.mat');

% setting priors
% i) prior for loading [alpha, beta]: N
loading_m0 = 0;
loading_s0 = 1;
% ii) prior for AR parameters [pi_i, pi_w]: N
ar_m0 = 0;
ar_s0 = 0.5;
% iii) prior for GARCH parameters [delt_bi, delt_ci, delt_bw, delt_cw]: IG
garch_a0 = 5;
garch_b0 = 1;
% iv) prior for varaince [eta]: IG
eta_a0 = 7;
eta_b0 = 4;

iter = 12000;
burn = 10000;
nacc = 0; % # total draws
out1 = zeros(iter-burn,33); % reserve memory
out2 = zeros(502, 12, iter-burn);
out3 = zeros(502, 6, iter-burn);

%out = post_mode';

par_old = mode';
post_old = -post_val;
%[HU,HS,HV] = svd(inv(post_hess));
%S = HU * sqrt(HS)*HV';
S = chol(inv(post_hess));


for ii = 1:iter
    % 1. propose new par
    par_new = par_old + (randn(1,33)*S)';
    
    % 2. evaluate posterior at new draw
    [alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par_new');
    % check parameter restrictions
    check = sum(abs(pi_i) > 1) + sum(abs(pi_w) >1) + sum(delt_bw < 0) + sum(delt_bw > 1) + sum(delt_cw < 0) + ...
        sum(delt_cw > 1) + sum(delt_aw < 0) + sum(delt_bi < 0) + sum(delt_bi > 1) + sum(delt_ci < 0) + ...
        sum(delt_ci > 1) + sum(delt_ai < 0) + sum(var_eta < 0 );
    if check == 0
        
        % calculate posterior
        post_new = -neg_posterior(par_new',data,loading_m0,loading_s0,ar_m0,ar_s0,garch_a0,garch_b0,eta_a0,eta_b0,0);
        
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
        %out = [out; par_new'];
    end
    
    acc_rate = nacc/ii;
    % lerning rate scheduling
    if ii > 500 && ii < 1500  
      if acc_rate > 0.4
          S = S * 1.01;
      elseif acc_rate<0.2
          S = S * 0.99;
      end
    end
      
    % store results
    if ii > burn
        out1(ii-burn,:) = par_old';
        
        % filtering
        % initialization of Kalman Filter
        P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
        [PU,PS,PV] = svd(P_0);
        omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

        [omega, ~, cond_var, ~] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,1);
        out2(:,:,ii-burn) = omega;
        out3(:,:,ii-burn) = cond_var;
    end

    if mod(ii, 100) == 0
        fprintf('Iter: %g \r', ii)
        disp(nacc)
        disp(acc_prob)
        %disp(check)
    end
    
end
save('out1', 'out1');
save('out2', 'out2');
save('out3', 'out3');
disp(acc_rate)
%% 4. plot
load('out1.mat')
load('out2.mat')
load('out3.mat')
load('true_par.mat')

% parameters
figure1 = figure;
foo = 1;
for n = 1:33
    subplot(5,7,foo);
    hist(out1(:,n), 50);
    title("true parameter " + true_par(n) + " ");
    foo = foo + 1; 
end

figure2 = figure;
foo = 1;
for n = 1:33
    subplot(5,7,foo);
    plot(out1(:,n));
    title("true parameter " + true_par(n) + " ");
    foo = foo + 1; 
end

% states
states_est = median(out2,3);
load('R_i.mat')
load('R_w.mat')
figure3 = figure;
foo = 1;
subplot(2,3,foo);
plot([states_est(:,6),R_w]);
legend('median estimate', 'true');title('common factor'); axis tight;
for n = 1:5
    foo = foo + 1; 
    subplot(2,3,foo);
    plot([states_est(:,n),R_i(:,n)]);
    legend('median estimate', 'true');title("ideosyncratic factor of country " + n + " "); axis tight;
end

% conditional variances
condvar_est = median(out3,3);
load('h_i.mat')
load('h_w.mat')
figure4 = figure;
foo = 1;
subplot(2,3,foo);
plot([condvar_est(:,6),h_w]);
legend('median estimate', 'true');title('conditional variances of common factor'); axis tight;
for n = 1:5
    foo = foo + 1; 
    subplot(2,3,foo);
    plot([condvar_est(:,n),h_i(:,n)]);
    legend('median estimate', 'true');title("conditional variances of ideosyncratic factor of country " + n + " "); axis tight;
end

saveas(figure1, 'par_hist.pdf')
saveas(figure2, 'par_plot.pdf')
saveas(figure3, 'states.pdf')
saveas(figure4, 'cond_var.pdf')