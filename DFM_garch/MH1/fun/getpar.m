function [alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(xopt)
%getpar get parameters
%   Detailed explanation goes here
N = 5;
pi_w    = xopt(1);
delt_bw = xopt(2);
delt_cw = xopt(3);
delt_aw = 1 - delt_bw - delt_cw;

pi_i    = xopt(4:8);
delt_bi = xopt(9:13);
delt_ci = xopt(14:18);
delt_ai = 1 - delt_bi - delt_ci;

var_eta = xopt(19:23);
alpha   = xopt(24:28);
beta    = xopt(29:33);

% sign restrictions
for nn = 1:N
    if alpha(nn) < 0
        alpha(nn) = -alpha(nn);
    end
end
if beta(1) < 0
   beta = -beta; 
end

end

