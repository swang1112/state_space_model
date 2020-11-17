function [alpha, P, v, F] = local_level_filter(y, alpha_0, P_0, sigma_obs, sigma_trans)
%local_level_filter: Kalman filter recursion for local level filter
%   y: Tx1 observable
%   alpha_0: intial state
%   P_0: initial state var
%   sigma_obs: error variance in the observation equation
%   signa_trans: error variance in the transition equation
    Tob = length(y);
    alpha = zeros(Tob,1);
    P = zeros(Tob,1);
    v = zeros(Tob,1);
    F = zeros(Tob,1);
    
    % initialization
    alpha(1) = alpha_0;
    P(1) = P_0;
    
    for t = 2:Tob
        alpha_tmp = alpha(t-1);
        P_tmp = P(t-1) + sigma_trans;
        v(t) = y(t) - alpha_tmp;
        F(t) = P_tmp + sigma_obs;
        
        alpha(t) = alpha_tmp + v(t)*P_tmp/F(t);
        P(t) = P_tmp - P_tmp*P_tmp/F(t) + sigma_trans;
    end
            
end
