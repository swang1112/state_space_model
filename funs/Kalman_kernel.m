function [alpha, P, LogLike, pred_error, pred_error_cov] = Kalman_kernel(y, alpha_0, P_0, A, X, H, F, Mu, Sigma_obs, Sigma_trans, Smooth)
%   y:              Txn matrix, data matrix of obs
%   alpha_0:        rx1 vector, init state 
%   P_0:            rxr matrix, init state conditional covar matrix
%   A:              nxk matrix, coeff matrix for exo obs
%   X:              Txk matrix, data matrix of exo obs
%   H:              nxr matrix, state loading
%   F:              rxr matrix, AR coeff matrix in the trans equation
%   Mu:             rx1 matrix, intercept in the trans equation
%   Sigma_obs:      nxn matrix, error covar matrix in the obs equation
%   Sigma_trans:    rxr matrix, error covar matrix in the trans equation
%   Smooth:         bool, if Smooth=true, smoothed estimates of states by backwards
%                   iterationss are returned

    [Tob, n]    = size(y);
    k           = size(X, 2);
    r           = size(H, 2);

    % storage reservation
    alpha           = zeros(Tob,r);             
    P               = zeros(r,r,Tob);              
    pred_error      = zeros(Tob,n);               % prediction error
    pred_error_cov  = repmat(eye(n), [1,1,Tob]);  % prediction error covariance
    
    if Smooth == 1
        P_10s           = repmat(eye(r), [1,1,Tob]);
        alpha_10s       = zeros(Tob,r);      
    end

    % init
    alpha(1,:)  = alpha_0';
    P(:,:,1)    = P_0;
    LogLike     = 0;

    for t = 2:Tob;
        % prediction state
        alpha_10 = Mu + F * alpha(t-1,:)';
        P_10     = F * P(:,:,t-1) * F' + Sigma_trans;
    
        % prediction obs
        v_10 = y(t,:)' - H * alpha_10 - A * X(t,:)';
        f_10 = H * P_10 * H' + Sigma_obs;
        
        % Log-likelihood 
        LogLike = LogLike - (n/2)*log(2*pi) - 1/2*(log(det(f_10)) + v_10'*inv(f_10)*v_10);
    
        % update
        K        = P_10 * H' * inv(f_10);
        alpha_11 = alpha_10 + K * v_10;
        P_11     = P_10 - K * H * P_10; 
    
        % store
        pred_error(t,:)       = v_10';
        pred_error_cov(:,:,t) = f_10;
        alpha(t,:)            = alpha_11';
        P(:,:,t)              = P_11;
        
        if Smooth == 1
            P_10s(:,:,t)    = P_10;
            alpha_10s(t,:)  = alpha_10';
        end
    
    end

    % smoothing
    if Smooth == 1
        % copy results from filtering
        alpha_smo   = alpha;
        P_smo       = P;
        for t = Tob-1:-1:1
            J               = P(:,:,t) * F' * inv(P_10s(:,:,t+1));
            alpha_smo(t,:)  = alpha(t,:) +  (alpha_smo(t+1,:) - alpha_10s(t+1,:)) * J';
            P_smo(:,:,t)    = P(:,:,t) +  J * (P_smo(:,:,t+1) - P_10s(:,:,t+1)) * J';
        end
        % replace filtering results
        alpha   = alpha_smo;
        P       = P_smo;
    end
    

end
