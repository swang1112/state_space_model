clc
clear

rng(37073)
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
x0(19:23) = 1e-4;
warning('off');
options = optimoptions(@fmincon, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 50000);
xopt = fmincon(@obj, x0, A, b, [],[],[],[],[], options);


%% objective function
function ll = obj(par)

pi_w    = par(1);
delt_bw = par(2);
delt_cw = par(3);
delt_aw = 1 - delt_bw - delt_cw;

pi_i    = par(4:8);
delt_bi = par(9:13);
delt_ci = par(14:18);
delt_ai = 1 - delt_bi - delt_ci;

var_eta = par(19:23);
alpha   = par(24:28);
beta    = par(29:33);

% sign restrictions
for nn = 1:5
    if alpha(nn) < 0
        alpha(nn) = -alpha(nn);
    end
end

if beta(1) < 0
   beta = -beta; 
end

% initialization of Kalman Filter
P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
[PU,PS,PV] = svd(P_0);
rng(37073); 
omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

load('data')
[~, ~, ~, ll] = kalman_garch(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta);
   ll = -ll;
end


