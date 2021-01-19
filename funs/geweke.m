function [z, pval] = geweke(x, Teil1, Teil2)
% x: R x 1, R is the # samples
% Teil1-2: [0, 1] 

R = size(x);
R = R(1);

N_1 = ceil(Teil1*R);
N_2 = ceil(Teil2*R);

Mean_1 = mean(x(1:N_1));
Mean_2 = mean(x(R-N_2+1:end));

Se_1 = dspectrum(x(1:N_1), 0);
Se_2 = dspectrum(x(R-N_2+1:end), 0);

z = (Mean_1-Mean_2)/sqrt((Se_1/N_1)+(Se_2/N_2));
pval = 2 * (1 - cdf('Normal',abs(z),0,1));
end