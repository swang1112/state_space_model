%%
init_post
%%
clear
clc
addpath('fun');
load('data.mat');
load('mode.mat');
load('post_val.mat');
load('post_hess.mat');

% setting priors
% i) prior for [alpha'; beta']: gaussian
loading_m0 = 0;
loading_s0 = 1;
% ii) prior for [pi_i'; pi_w]: gaussian
ar_m0 = 0;
ar_s0 = 0.5;
% iii) prior for [delt_bi'; delt_ci'; delt_bw; delt_cw]: IG
garch_a0 = 5;
garch_b0 = 1;
% iv) prior for [eta]: IG
eta_a0 = 7;
eta_b0 = 4;

iter = 8000;
burn = 5000;
nacc = 0; % # total draws
out = zeros(iter-burn,33); % reserve memory
%out = post_mode';

par_old = mode';
post_old = -post_val;
%[HU,HS,HV] = svd(inv(post_hess));
%S = HU * sqrt(HS)*HV';
%S = diag(sqrt(diag(inv(post_hess))));
S = chol(inv(post_hess));

%%
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
        % initialization of Kalman Filter
        
        post_new = -neg_posterior(par_new',data,loading_m0,loading_s0,ar_m0,ar_s0,garch_a0,garch_b0,eta_a0,eta_b0);
    
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
          S = S * 1.001;
      elseif acc_rate<0.2
          S = S * 0.99;
      end
    end
      
    % store results
    if ii > burn
        out(ii-burn,:) = par_old';
    end

    if mod(ii, 100) == 0
        fprintf('Iter: %g \r', ii)
        disp(nacc)
        disp(acc_prob)
        %disp(check)
    end
    
end
save('out', 'out');
disp(acc_rate)
%% print and plot
load('out.mat')
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
% pi_i    = [1e-6, 0.035, -0.075, -0.092, 0.044];
