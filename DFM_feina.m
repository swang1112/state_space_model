%% DFM
% common factor: G_t
% series specific factor: eps_jt

% y_jt = phi_j * G_t + eps_jt
% G_t = rho * G_t-1 + v_t, v_t ~ N(0, sigma_v)                   
% eps_jt = gamma_j * eps_jt + ets_jt, ets_jt ~ N(0, sigma_eta_j);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% identification restriction
% DFM 1 in Bai and Wang (2012)
% 1) sigma_v = 1
% 2) top rxr matrix of phi is lower-triangular matrix and its diagonal elements are strictly positive 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc; 

%% Data simulation
rng(1234); 
T = 200; 
N = 15; 
 
% common factor
rho = rand(1, 1); % AR coefficient 
% identification restriction
phi = [rand(1, 1); randn(N-1, 1)]; % factor loadings
sigma_v = 1; % rand(1,1); % variance of error term 

% series specific factors
gamma = rand(N, 1)*0.95; % AR coefficient 
sigma_eta = rand(N, 1); % variance of error term

theta_true = [phi; rho; gamma; sigma_eta]; 

[y, G, eps] = DFM_simul(T, N, rho, phi, gamma, sigma_eta, sigma_v); 
save y.mat

%% ML
theta0 = zeros(3*N+1,1); 
options = optimoptions('fminunc','Display','iter');
theta_opt = fminunc(@Maximand, theta0, options);

%% Kalman Filter
theta = theta_opt;

phi = theta(1:N,:); 
phi(1) = abs(phi(1))+0.0000001; 

theta(N+1:2*N+1,:) = theta(N+1:2*N+1,:)./(ones(N+1,1)+abs(theta(N+1:2*N+1,:))); % stationary G and eps
theta(2*N+2: end, :) = exp(theta(2*N+2: end, :)); % positive variances


rho = theta(N+1, :); 
gamma = theta(N+2:2*N+1,:); 
sigma_eta = theta(2*N+2:3*N+1,:);  
sigma_v = 1;

theta_est = [phi; rho; gamma; sigma_eta]; 

H = [phi eye(N)];  
F = diag([rho; gamma]); 
Q = diag([sigma_v; sigma_eta]); 
R = eye(N)*0.000001; 

y_data = y'; 
x_data = []; 
mu = zeros(N+1,1); 
A = []; 

beta_00 = zeros(N+1, 1); 
P_00 = eye(N+1)*1000;
P_00(1,1) = 1000; 
for j = 1:N
    P_00(j+1, j+1) = sigma_eta(j)/(1-gamma(j)^2); 
end


[beta_tt, beta_tT, ML] = kalman(beta_00, P_00, A, H, R, Q, F, mu, y_data, x_data, N+1); 
beta = squeeze(beta_tt)'; 
figure; plot(y); title('y'); 
figure;  plot(G, 'LineWidth', 1.5); hold on; plot(beta(:,1), 'LineWidth', 1.5); legend('true common factor','estimated common factor');title('Common Factor'); 
figure; 
for j = 1:N
    subplot(3,5,j);plot(eps(:,j), 'LineWidth', 1.5); hold on;  plot(beta(:,j+1), 'LineWidth', 1.5); legend('true', 'estimated'); title(sprintf('%2d th series specific factor', j)); 
end
 
names1 = strings(N, 1); names2 = strings(N, 1); names3 = strings(N, 1); names4 = strings(N, 1); 
for j = 1:N
    names1(j) = "phi" + num2str(j);
    names2(j) = "gamma" + num2str(j);
    names3(j) = "sigma_eta" + num2str(j);
end
names = [names1; "rho"; names2; names3]; 
table(names, theta_true, theta_est)

ML

%% R
R = zeros(N, 1);
for j = 1:N
    R(j) = (theta(j)^2 * var(beta(:,1)))/ (theta(j)^2 * var(beta(:,1)) + var(beta(:,j+1)));
end
figure; bar(R);ylim([0,1]);  title('Fraction of variance due to common factor R');
R

%% Estimation
function LL = Maximand(theta)

data = load('y.mat');
y = data.y; 
[T, N] = size(y); 

phi = theta(1:N,:);
phi(1) = abs(phi(1))+0.0000001; 

% theta = [phi1...phiN; rho0; gamma10...gammaN0; sigma_eta10....sigma_etaN0; sigma_v0 ]
theta(N+1:2*N+1,:) = theta(N+1:2*N+1,:)./(ones(N+1,1)+abs(theta(N+1:2*N+1,:))); % stationary G and eps
theta(2*N+2: end, :) = exp(theta(2*N+2: end, :)); % positive variances


rho = theta(N+1, :); 
gamma = theta(N+2:2*N+1,:); 
sigma_eta = theta(2*N+2:3*N+1,:);  
sigma_v = 1;

H = [phi eye(N)];  
F = diag([rho; gamma]); 
Q = diag([sigma_v;sigma_eta]);
R = eye(N)*0.0000001; 

y_data = y'; 
x_data = []; 
mu = zeros(N+1,1); 
A = []; 

beta_00 = zeros(N+1, 1); 
P_00 = eye(N+1)*1000;
P_00(1,1) = sigma_v/(1-rho^2); 
for j = 1:N
    P_00(j+1, j+1) = sigma_eta(j)/(1-gamma(j)^2); 
end


[~,~,LL_cum] = kalman(beta_00, P_00, A, H, R, Q, F, mu, y_data, x_data, N+1); 

LL = -LL_cum; 

end




















