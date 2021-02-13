clear all;
close all; 
clc;
rng(1234);
%% parameters
Tob     = 502;
burn    = 10;
N       = 5;

% Table 2 Berger&Pozzi(2013)
pi_w    = 0.126;
delt_aw = 0.086;
delt_bw = 0.115;
delt_cw = 0.799;

pi_i    = [1e-6, 0.035, -0.075, -0.092, 0.044];
delt_ai = [0.022, 0.001, 0.007, 0.012, 0.026];
delt_bi = [0.133, 0.130, 0.233, 0.089, 0.067];
delt_ci = [0.845, 0.869, 0.760, 0.900, 0.906];

%var_eta = [3.34e-4, 3.42e-4, 3.34e-4, 3.41e-4, 3.34e-4];
var_eta = [3.34e-2, 3.42e-2, 3.34e-2, 3.41e-2, 3.34e-2];
alpha   = [3.317, 3.430, 5.692, 3.695, 4.609];
beta    = [4.887, 4.501, 3.442, 3.282, 2.766];

true_par = [pi_w, delt_bw, delt_cw, pi_i, delt_bi, delt_ci, var_eta, alpha, beta];
save('true_par', 'true_par');
%% 
% memory reservation
TT = Tob + burn;
data    = zeros(TT, N);

% commmon factor
eps_w   = zeros(TT,1);
h_w     = zeros(TT,1);
R_w     = zeros(TT,1);

% country sepcific
eps_i   = zeros(TT, N);
h_i     = zeros(TT, N);
R_i     = zeros(TT, N);

% initial values
h_w(1)   = 1;
h_i(1,:) = 1;

for t = 2:TT;
    h_w(t)   = delt_aw + delt_bw * (eps_w(t-1))^2 + delt_cw * h_w(t-1);
    eps_w(t) = sqrt(h_w(t)) * randn(1);
    R_w(t)   = pi_w * R_w(t-1) + eps_w(t);
    
    for n = 1:N
       h_i(t,n)   = delt_ai(n) + delt_bi(n) * (eps_i(t-1,n))^2 + delt_ci(n) * h_i(t-1,n);
       eps_i(t,n) = sqrt(h_i(t,n)) * randn(1);
       R_i(t,n)   = pi_i(n) * R_i(t-1,n) + eps_i(t,n);
    end
    
    for n = 1:N
        data(t, n) = alpha(n) * R_i(t,n) + beta(n) * R_w(t) + randn(1)*sqrt(var_eta(n));
    end        
end

%%
R_w = R_w((burn+1):end, :);
save('R_w', 'R_w');
R_i = R_i((burn+1):end, :);
save('R_i', 'R_i');
data = data((burn+1):end, :);
save('data', 'data');
h_w = h_w((burn+1):end, :);
save('h_w', 'h_w');
h_i = h_i((burn+1):end, :);
save('h_i', 'h_i');
