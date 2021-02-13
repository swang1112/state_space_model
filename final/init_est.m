clc
clear
%addpath('sims_Optimization')
rng(37073)

% init optimizer
load('xopt.mat');
load('data.mat');

x0 = xopt;
x0(21) = x0(19);
options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 50000);
%[post_val,mode,gh,post_hess,itct,fcount,retcodeh] = csminwel(@obj,x0,eye(length(x0))*.1,[],1e-15,20000,data);
[mode,post_val,~,~,~,post_hess] = fminunc(@obj,x0,options,data);

save('mode', 'mode');
save('post_val', 'post_val')
save('post_hess', 'post_hess');


%% posterior
function out = obj(par, data)
%par = par';
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
eta_d0 = 0.2;

out = neg_posterior(par,data,loading0,loading0s,ar0,ar0s,garch_v0,garch_d0,eta_v0,eta_d0);

end

