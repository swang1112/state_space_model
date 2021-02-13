%%
clear
clc
addpath('fun');
load('data.mat');
load('mode');
load('post_val');
load('post_hess');
iter = 4000;
burn = 3000;
nacc = 0; % # total draws
out = zeros(iter-burn,33); % reserve memory
%out = post_mode';

par_old = mode';
post_old = -obj(par_old);
[HU,HS,HV] = svd(inv(post_hess));
S = HU * sqrt(HS)*HV';
%%
for ii = 1:iter
    % 1. propose new par
    par_new = par_old + (randn(1,33)*S)';
    
    % 2. evaluate posterior at new draw
    [alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par_new');
    % check parameter restrictions
    check = sum(abs(pi_i) > 1) + sum(abs(pi_w) >1) + sum(delt_bw < 0) + sum(delt_bw > 1) +...
        sum(delt_cw < 0) + sum(delt_cw > 1) + sum(delt_aw < 0) + sum(delt_bi < 0) + ...
        sum(delt_bi > 1) + sum(delt_ci < 0) + sum(delt_ci > 1) + sum(delt_ai < 0) + sum(var_eta < 0 );
    if check == 0
        % initialization of Kalman Filter
        
        post_new = -obj(par_new);
    
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
    %if ii > 500 && ii < 1500  
      if acc_rate > 0.4
          S = S * 1.0000001;
      elseif acc_rate<0.2
          S = S * 0.99;
      end
    %end
      
    % store results
    if ii > burn
        out(ii-burn,:) = par_old';
    end

    if mod(ii, 100) == 0
        fprintf('Iter: %g \r', ii)
        disp(nacc)
        disp(check)
    end
    
end
%out = out(burn:end,:);
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
garch_d0 = 0.5;
% iv) prior for var_eta: IG
eta_v0 = 5;
eta_d0 = 0.1;

out = neg_posterior(par,data,loading0,loading0s,ar0,ar0s,garch_v0,garch_d0,eta_v0,eta_d0);
end