function out = neg_posterior(par,data,loading_m0,loading_s0,ar_m0,ar_s0,garch_a0,garch_b0,eta_a0,eta_b0)
%calculate -1*posterior  
[alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par);

%check = sum(abs(pi_i) > 1) + sum(abs(pi_w) >1) + sum(delt_bw < 0) + sum(delt_bw > 1) + sum(delt_cw < 0) + sum(delt_cw > 1) + sum(delt_aw < 0) + sum(delt_bi < 0) + sum(delt_bi > 1) + sum(delt_ci < 0) + sum(delt_ci > 1) + sum(delt_ai < 0) + sum(var_eta < 0 );
%if check == 0
    % initialization of Kalman Filter
    P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
    [PU,PS,PV] = svd(P_0);
    omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

    [~, ~, ~, ll] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);
    
    foo = ll;
    for j = 1:5
        foo = foo + lgam1(1/delt_bi(j), garch_a0, garch_b0) + lgam1(1/delt_ci(j), garch_a0, garch_b0) + ...
            lgam1(1/var_eta(j), eta_a0, eta_b0) + log(normpdf(alpha(j), loading_m0, loading_s0)) + ...
            log(normpdf(beta(j), loading_m0, loading_s0)) + log(normpdf(pi_i(j), ar_m0, ar_s0)) ;
    end
    out = foo + log(normpdf(pi_w, ar_m0, ar_s0)) + lgam1(1/delt_bw, garch_a0, garch_b0) + lgam1(1/delt_cw, garch_a0, garch_b0);
    out = -out;
%else
%    out = inf;
%end

end

