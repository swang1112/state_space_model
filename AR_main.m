%% DGP 
clc
clear
rng(37077);
addpath('funs');

Tob     = 200; % total observation
mu      = 0.2;
phi     = 0.9;
sigma2  = 1;

y = zeros(Tob,1);
for t = 2:Tob
    y(t) = mu + phi * y(t-1) + sqrt(sigma2)*randn(1);
end

%plot(y);
%% Gibbs sampling
% model representation
T   = Tob - 1;
X   = [ones(T, 1) y(1:Tob-1)];
Y   = y(2:end);


iter    = 5000;
burn    = iter - 1000;

% prior for beta = [mu; phi]: N(beta0, beta_s0)
phi_0   = 0.9;
mu_0    = mean(y)*(1-phi_0);
beta0   = [mu_0;phi_0];
beta_s0 = [0.2, 0; 0, 0.1];

% prior for sigma2: IG(0.5v0, 0.5delta0) 
v0      = 1;
delta0  = 0.1;

% starting values
b0      = inv(X'*X)*X'*Y;
Sigma2  = (Y - X*b0)'*(Y - X*b0)/(T - 2);

% memory reservation
Betas   = zeros(1000, 2);
Sigma2s = zeros(1000, 1);

for i = 1:iter
    % 1. beta|sigma2 
    beta1   = inv(inv(beta_s0) + (1/Sigma2) * X'*X) * (inv(beta_s0) * beta0 + (1/Sigma2) * X'*Y);
    beta_s1 = inv(inv(beta_s0) + (1/Sigma2) * X'*X);
    
    % draw under stationarity-restriction
    foo     = 0;
    while foo == 0
        Beta    = beta1 + chol(beta_s1)'*randn(2,1);
        if abs(Beta(2)) < 1
            foo = 1;
        end
    end
    
    % 2. sigma2|beta
    v1      = v0 + T;
    delta1  = delta0 + (Y - X*Beta)'*(Y - X*Beta);
    
    % draw
    faa     = randn(v1, 1);
    Sigma2  = delta1 / (faa' * faa);
    
    % store the results
    if i > burn
        Betas(i-burn,:)   = Beta';
        Sigma2s(i-burn,:) = Sigma2;
    end
    
end

%% print and plot
[mu phi]
mean(Betas)
prctile(Betas, [5, 95])

sigma2
mean(Sigma2s)
prctile(Sigma2s, [5, 95])


subplot(3,3,1);
hist(Betas(:,1), 50);
title("intercept (true: " + mu + " )");
subplot(3,3,2);
plot(Betas(:,1));
xlabel("Iter");
subplot(3,3,3);
autocorr(Betas(:,1));
subplot(3,3,4);
hist(Betas(:,2), 50);
title("AR coeff (true: " + phi + " )");
subplot(3,3,5);
plot(Betas(:,2));
xlabel("Iter");
subplot(3,3,6);
autocorr(Betas(:,2));
subplot(3,3,7);
hist(Sigma2s, 50);
title("Fehlervarianz (true: " + sigma2 + " )");
subplot(3,3,8);
plot(Sigma2s);
xlabel("Iter");
subplot(3,3,9);
autocorr(Sigma2s);

%% Geweke test
[z_1, pval_1] = geweke(Betas(:,1), 0.1, 0.5)
[z_2, pval_2] = geweke(Betas(:,2), 0.1, 0.5)
[z_3, pval_3] = geweke(Sigma2s, 0.1, 0.5) 

