function [beta_ttm1,P_ttm1, beta_tt,P_tt, beta_tT,P_tT, J_t, f_ttm1, ita_ttm1] = kalman(beta_00, P_00, A, H, R, Q, F, mu, y_data, x_data, n_state_var); 
 %function [beta_ttm1,P_ttm1, beta_tt,P_tt, f_ttm1, ita_ttm1] = kalman(beta_00, P_00, A, H, R, Q, F, mu, y_data, x_data, n_state_var); 
[n,T] = size(y_data); 

beta_ttm1     = NaN(n_state_var,1,T);
beta_tt          = NaN(n_state_var,1,T);

P_ttm1         = NaN(n_state_var,n_state_var,T);
P_tt               = NaN(n_state_var,n_state_var,T); 

f_ttm1          = NaN(n,n,T); 
ita_ttm1       = NaN(n,1,T);

LL_vec = zeros(1,T); 

J_t                 = NaN(n_state_var,n_state_var,T); 

beta_tT          = NaN(n_state_var,1,T);
P_tT               = NaN(n_state_var,n_state_var,T);

beta_tt(:,:,1)  = beta_00;
P_tt(:,:,1)       = P_00;


%%%% filtering 
for i = 2:1:T
% prediction
beta_ttm1(:,:,i)  =  F*beta_tt(:,:,i-1) + mu(:,:,i); 
P_ttm1(:,:,i)     = F*P_tt(:,:,i-1)*F' + Q ;
    

% prediction error
ita_ttm1(:,:,i) = y_data(:,i) - H*beta_ttm1(:,:,i) - A*x_data(:,i); 
f_ttm1(:,:,i)   = H*P_ttm1(:,:,i)*H' + R;


% LL
LL_vec(i) = -(n / 2) * log(2 * pi ) - 0.5 * log(det(f_ttm1(:,:,i))) -0.5 * ita_ttm1(:,:,i)' * inv(f_ttm1(:,:,i))* ita_ttm1(:,:,i);


%update      
beta_tt(:,:,i) = beta_ttm1(:,:,i) + P_ttm1(:,:,i)*H'*inv(f_ttm1(:,:,i))*ita_ttm1(:,:,i); 
P_tt(:,:,i)    = P_ttm1(:,:,i) - P_ttm1(:,:,i)*H'*inv(f_ttm1(:,:,i))*H*P_ttm1(:,:,i);       
 end; 
        
 LL_cum = sum(LL_vec);

for k = T-1:-1:1; 
J_t(:,:,k)      = P_tt(:,:,k)*F'*inv(P_ttm1(:,:,k+1)); 
end;


beta_tT(:,:,T) =       beta_tt(:,:,T); 
P_tT(:,:,T)      =      P_tt(:,:,T); 


%%%% smoothing         
for j = T-1:-1:1
beta_tT(:,:,j)  = beta_tt(:,:,j) + J_t(:,:,j)*(beta_tT(:,:,j+1) - beta_ttm1(:,:,j+1));    
P_tT(:,:,j)     = P_tt(:,:,j) + J_t(:,:,j)*(P_tT(:,:,j+1) - P_ttm1(:,:,j+1))*J_t(:,:,j)'; 
end; 



end












