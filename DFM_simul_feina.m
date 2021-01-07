%% Data simulation
function[y, G, eps] = DFM_simul_feina(T, N, rho, phi, gamma, sigma_eta, sigma_v);  

T = T+50; 

% common factor 
G = zeros(T, 1);
for i = 2:T
    G(i) = rho * G(i-1) + randn(1,1) * sqrt(sigma_v);
end


% series specific factors
y = ones(T, N); 
eps = ones(T, N);

for j = 1:N
    for i = 2:T
        eps(i, j) = gamma(j) * eps(i-1, j) + randn(1,1) * sqrt(sigma_eta(j)); 
    end
    y(:, j) = phi(j) * G + eps(:, j); 
end

y = y(51:end,:);
eps = eps(51:end,:); 
G = G(51:end,:); 


end