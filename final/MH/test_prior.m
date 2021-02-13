x = linspace(-0.1, 5, 2000);
y = exp(lgam1(1./x,10,1));
plot(x,y)
lgam1(1./3.35e-2,7,4)
%x = linspace(-5, 5, 1000);
%y = normpdf(x,0,0.5);
%plot(x,y)
%%
x = linspace(-0.5, 1, 2000);
y = flatprior(x, 0.5, -6, 6);
plot(x,y)


%%

% setting priors
% i) prior for [alpha'; beta']: gaussian
loading0 = zeros(10,1)+0.1;
loading0s= eye(10)*0.5;
% ii) prior for [pi_i'; pi_w]: gaussian
ar0 = zeros(6,1);
ar0s= eye(6)*0.5;
% iii) prior for [delt_bi'; delt_ci'; delt_bw; delt_cw]: IG
garch_v0 = 7;
garch_d0 = 1;
% iv) prior for var_eta: IG
eta_v0 = 7;
eta_d0 = 1;

%calculate -1*posterior  
[alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta] = getpar(x0);

check = sum(abs(pi_i) > 1) + sum(abs(pi_w) >1) + sum(delt_bw < 0) + sum(delt_bw > 1) + sum(delt_cw < 0) + sum(delt_cw > 1) + sum(delt_aw < 0) + sum(delt_bi < 0) + sum(delt_bi > 1) + sum(delt_ci < 0) + sum(delt_ci > 1) + sum(delt_ai < 0) + sum(var_eta < 0 );
if check == 0
    % initialization of Kalman Filter
    P_0     = [diag([1./(1-pi_i.^2), 1/(1-pi_w^2)]), eye(6); eye(6), eye(6)];
    [PU,PS,PV] = svd(P_0);
    rng(37073); 
    omega_0 = randn(1,12) * PU * sqrt(PS)*PV';

    [~, ~, ~, ll] = kalman_garch_uni(data, omega_0, P_0, alpha, beta, pi_i, pi_w, delt_aw, delt_bw, delt_cw, delt_ai, delt_bi, delt_ci,var_eta,0);
    
    foo = ll;
    for j = 1:5
        foo = foo + lgam(garch_v0, garch_d0, 1./delt_bi(j)) + lgam(garch_v0, garch_d0, 1./delt_ci(j)) + lgam(eta_v0, eta_d0, 1./var_eta(j));
    end
    out = foo + log(mvnpdf([alpha'; beta'], loading0, loading0s)) + log(mvnpdf([pi_i'; pi_w'], ar0, ar0s)) + lgam(garch_v0, garch_d0, 1./delt_bw) +  lgam( garch_v0, garch_d0, 1./delt_cw);
    out = -out;
else
    out = inf;
end
    
    
    