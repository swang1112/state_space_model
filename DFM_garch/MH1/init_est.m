clc
clear
addpath('fun')
rng(37073)

% init optimizer
%load('xopt.mat');
load('data.mat');

% parameter restrictions
A = zeros(47,33);
b = zeros(47,1);

% 1 GARCH parameters
% 1.1 delta_b, delta_c > 0
A(1,2) = -1;
A(2,9) = -1;
A(3,10) = -1;
A(4,11) = -1;
A(5,12) = -1;
A(6,13) = -1;
A(7,3) = -1;
A(8,14) = -1;
A(9,15) = -1;
A(10,16) = -1;
A(11,17) = -1;
A(12,18) = -1;
b(1:12) = 0;

% 1.2 delta_b, delta_c < 1
A(13,2) = 1;
A(14,9) = 1;
A(15,10) = 1;
A(16,11) = 1;
A(17,12) = 1;
A(18,13) = 1;
A(19,3) = 1;
A(20,14) = 1;
A(21,15) = 1;
A(22,16) = 1;
A(23,17) = 1;
A(24,18) = 1;
b(13:24) = 1;

% 1.3 delta_b + delta_c < 1
A(25, [2,3]) = 1;
A(26, [9,14]) = 1;
A(27, [10,15]) = 1;
A(28, [11,16]) = 1;
A(29, [12,17]) = 1;
A(30, [13,18]) = 1;
b(25:30) = 1;

% 2 AR parameters
% 2.1 pi < 1
A(31,1) = 1;
A(32,4) = 1;
A(33,5) = 1;
A(34,6) = 1;
A(35,7) = 1;
A(36,8) = 1;
b(31:36) = 1;

% 2.2 pi > -1
A(37,1) = -1;
A(38,4) = -1;
A(39,5) = -1;
A(40,6) = -1;
A(41,7) = -1;
A(42,8) = -1;
b(37:42) = 1;

% 3 measurement error variances
% etas > 0
A(43,19) = -1;
A(44,20) = -1;
A(45,21) = -1;
A(46,22) = -1;
A(47,23) = -1;
b(43:47) = 0;

% init optimizer
x0 = rand(1,33)*0.8 + 0.1;
x0(19:23) = 1e-2;
warning('off');
options = optimoptions(@fmincon, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 50000);
[mode,post_val,~,~,~,~,post_hess] = fmincon(@obj, x0, A, b, [],[],[],[],[], options, data);

save('mode', 'mode');
save('post_val', 'post_val')
save('post_hess', 'post_hess');

%% posterior
function out = obj(par, data)
%par = x0;
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

out = neg_posterior(par,data,loading_m0,loading_s0,ar_m0,ar_s0,garch_a0,garch_b0,eta_a0,eta_b0,1);


end

