function out = neg_posterior(par,data,loading_m0,loading_s0,ar_m0,ar_s0,garch_a0,garch_b0,eta_a0,eta_b0,replica)
%calculate -1*posterior  
[alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(par);

    % init of Kalman Filter
    P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
    if replica == 1
        rng(37073); 
    end
    [PU,PS,PV] = svd(P_0);
    omega_0 = randn(1,12) * PU * sqrt(PS)*PV';
    % loglike
    [~, ~, ~, ll] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);
    
    % prior
    foo = ll;
    for j = 1:5
        foo = foo + log(gampdf(1/delt_bi(j), garch_a0, garch_b0)) + log(gampdf(1/delt_ci(j), garch_a0, garch_b0)) + ...
            log(gampdf(1/var_eta(j), eta_a0, eta_b0)) + log(normpdf(alpha(j), loading_m0, loading_s0)) + ...
            log(normpdf(beta(j), loading_m0, loading_s0)) + log(normpdf(pi_i(j), ar_m0, ar_s0)) ;
    end
    
    % posterior
    out = foo + log(normpdf(pi_w, ar_m0, ar_s0)) + log(gampdf(1/delt_bw, garch_a0, garch_b0)) + log(gampdf(1/delt_cw, garch_a0, garch_b0));
    out = -out;

end

