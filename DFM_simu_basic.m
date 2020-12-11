function [y, F, eps] = DFM_simu_basic(Tob, p_f, p_e, Phi_f, Phi_e, sigma_f, sigma_e, Loading, burn)
% Simulate basic DFM with both common and idiosyncratic factors follwing 
% AR(1) processes
% 
% ACTUNG: Factors are subjected to restrictions: E(F*F') is diagonal
%         Loading*Loading' = eye(r)!!
%
%
% Tob       int, number of total observations
% p_f 		int, number of lags in common factors
% p_e 		int, number of lags in idiosyncratic components
% Phi_f     rp_fxrp_f matrix, AR coeff matrix of common factors (companion form)
% Phi_e 	Np_exNp_e matrix, AR coeff matrix of idiosyncratic components (companion form)
% Loading   rp_fxrp_f matrix, factor loading
% sigma_f	rxr matrix, covariance matrix of erros in common factors
% sigma_e	NxN matrix, covariance matrix of erros in idiosyncratic components
% burn		int, number of observations to be discard


% number of common factors
[rp, ~] = size(Phi_f);
r 		= rp / p_f;

% number of idiosyncratic components
[Np, ~] = size(Phi_e);
N 		= Np / p_e;

% reserve storage
simus 	= Tob+burn;
y 		= zeros(simus, N);
F 		= zeros(simus, r*p_f);
eps 	= zeros(simus, N*r);

% initial value
F(1,:) 	= rand(1, r*p_f);
eps(1,:)= rand(1, N*p_e);

for t = 2:simus

	v_t   = chol(sigma_f)*randn(r,1);		
	eta_t = chol(sigma_e)*randn(N,1);

	if p_f > 1
		v_t = [v_t; zeros(N*(p_e - 1),1)];
    else
		v_t = v_t;
    end
    
    if p_e > 1
		eta_t = [eta_t; zeros(N*(p_e - 1),1)];
    else
		eta_t = eta_t;
    end



	F_t = Phi_f * F(t-1,:)' + v_t;
	E_t = Phi_e * eps(t-1,:)' + eta_t;
	y_t = Loading * F_t(1:r,:) + E_t(1:N,:);

	F(t,:) 		= F_t';
	eps(t,:) 	= E_t';
	y(t,:) 		= y_t';
end

y 	= y((burn+1):end, :);
F 	= F((burn+1):end, 1:r);
eps = eps((burn+1):end, 1:N);

end

