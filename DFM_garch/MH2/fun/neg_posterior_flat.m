function out = neg_posterior_flat(par,data,c, L_loading, U_loading, L_var, U_var, L_ar, U_ar, replica)
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
        foo = foo + log(flatprior(alpha(j), c, L_loading, U_loading)) + log(flatprior(beta(j), c, L_loading, U_loading)) + ...
            log(flatprior(delt_bi(j), c, L_var, U_var)) + log(flatprior(delt_ci(j), c, L_var, U_var)) + log(flatprior(var_eta(j), c, L_var, U_var)) +...
            log(flatprior(pi_i(j), c, L_ar, U_ar));
    end
    
    % posterior
    out = foo + log(flatprior(pi_w, c, L_ar, U_ar)) + log(flatprior(delt_bw, c, L_var, U_var)) + log(flatprior(delt_cw, c, L_var, U_var));
    out = -out;


end

