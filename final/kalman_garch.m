function [omega, P, cond_var, LogLike] = kalman_garch(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta)
%%
%   y:              matrix, data matrix of obs
%   omega_0:        vector, init state 
%   P_0:            matrix, init state conditional covar matrix
%   H:              matrix, state loading
%   F:              matrix, AR coeff matrix in the trans equation
%   Sigma_obs:      matrix, error covar matrix in the obs equation
%   Sigma_trans:    matrix, error covar matrix in the trans equation

y = data;
[Tob, N] = size(y);
H = [diag(alpha), beta', zeros(N,N+1)];
F = diag([pi_i, pi_w, zeros(1,N+1)]);
K = [eye(N+1), eye(N+1)]';
Sigma_obs = diag(var_eta);
Delt_ai = delt_ai';
Delt_bi = diag(delt_bi);
Delt_ci = diag(delt_ci);

% storage reservation
r               = (N+1)*2;
omega           = zeros(Tob,r);             
P               = zeros(r,r,Tob);              
pred_error      = zeros(Tob,N);              
pred_error_cov  = repmat(eye(N), [1,1,Tob]);  
cond_var        = zeros(Tob, N+1);

P_10s           = repmat(eye(r), [1,1,Tob]);
omega_10s       = zeros(Tob,r);      

% init
omega(1,:)  = omega_0;
P(:,:,1)    = P_0;
LogLike     = 0;
cond_var(1,:) = 1;

for t = 2:Tob;
        % prediction state
        omega_10 = F * omega(t-1,:)';
        cond_var(t,1:N) = (Delt_ai + Delt_bi*(omega(t-1,(N+2):(N+6))'.^2 + diag(P((N+2):(N+6),(N+2):(N+6),t-1))) + Delt_ci*cond_var(t-1,1:N)')';
            
        cond_var(t,N+1) = delt_aw + delt_bw*(omega(t-1,end)^2 + P(end,end,t-1)) + delt_cw*cond_var(t-1,N+1);
        Sigma_trans = diag(cond_var(t,:));
        P_10     = F * P(:,:,t-1) * F' + K*Sigma_trans*K';
    
        % prediction obs
        v_10 = y(t,:)' - H * omega_10;
        f_10 = H * P_10 * H' + Sigma_obs;
        
        % Log-likelihood 
        LogLike = LogLike - (N/2)*log(2*pi) - 1/2*(log(det(f_10)) + v_10'*inv(f_10)*v_10);
    
        % update
        KG       = P_10 * H' * inv(f_10);
        omega_11 = omega_10 + KG * v_10;
        P_11     = P_10 - KG * H * P_10; 
          
        % store
        pred_error(t,:)       = v_10';
        pred_error_cov(:,:,t) = f_10;
        omega(t,:)            = omega_11';
        P(:,:,t)              = P_11;
        
        % for smoothing
            P_10s(:,:,t)    = P_10;
            omega_10s(t,:)  = omega_10';
    
    end
%%
    % smoothing
        % copy results from filtering
        omega_smo   = omega;
        P_smo       = P;
        for t = Tob-1:-1:1
            J               = P(:,:,t) * F' * inv(P_10s(:,:,t+1));
            omega_smo(t,:)  = omega(t,:) +  (omega_smo(t+1,:) - omega_10s(t+1,:)) * J';
            P_smo(:,:,t)    = P(:,:,t) +  J * (P_smo(:,:,t+1) - P_10s(:,:,t+1)) * J';
        end
        % replace filtering results
        omega   = omega_smo;
        P       = P_smo;
        

end