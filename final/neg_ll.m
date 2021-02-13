function out = neg_ll(par,data)
%calculate -1*posterior  
[alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par);

%check = sum(abs(pi_i) > 1) + sum(abs(pi_w) >1) + sum(delt_bw < 0) + sum(delt_bw > 1) + sum(delt_cw < 0) + ...
%sum(delt_cw > 1) + sum(delt_aw < 0) + sum(delt_bi < 0) + sum(delt_bi > 1) + sum(delt_ci < 0) + sum(delt_ci > 1) + ...
%    sum(delt_ai < 0) + sum(var_eta < 0 );
%if check == 0
    % initialization of Kalman Filter
    P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
    [PU,PS,PV] = svd(P_0);
    rng(37073); 
    omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

    [~, ~, ~, ll] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);
    
    out = -ll;
%else
%    out = inf;
%end

end

