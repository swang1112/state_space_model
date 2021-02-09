function out = lpdf_gauss(x, mu, sigma)
%log density of gaussian
[n,~] = size(x);
out   = log(1/(2*(pi^n/2))) -0.5*log(det(sigma)) - 0.5*(x - mu)'*inv(sigma)*(x - mu);
end

