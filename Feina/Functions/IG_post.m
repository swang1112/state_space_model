function[var_e] = IG_post(c0, C0, res)

T = size(res, 1); 

c1               = c0 + T/2;   % posterior shape
C1              = C0 + res'*res/2;  % posterior scale

var_e = inv(inv(C1)*gamrnd(c1,1,1,1)); 
% var_e = 1./gamrnd(c1,1/C1);