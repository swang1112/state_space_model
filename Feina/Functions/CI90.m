function[CI90] = CI90(a, b, T, MC, ind)
N = size(a, 1); 
if ind == 1; 
    % Normal distribution
    % a = b0; b = B0; (vector Nx1)
    y = zeros(N, MC); 
    for i = 1:MC
        for j = 1:N
        y(j,i) = a(j) + chol(b(j))*randn(1,1); 
        end
    end
    CI90 = [prctile(y, 5,2) prctile(y, 95, 2)]; 
       
    else if ind == 2; 
    % IG
    % a = bel; b = str;                 % prior strength
    c0          = b * T;                    % prior shape 
    C0         = c0 .* a;                  % prior scale 
    S_sigma2 = zeros(N, MC); 
    
    for i = 1:MC
        for j = 1:N
        S_sigma2(j, i) = inv(inv(C0(j))*gamrnd(c0(j),1,1,1)); 
        end
    end
    CI90 = [prctile(S_sigma2, 5, 2) prctile(S_sigma2, 95, 2)]; 
    
        end
end
    

