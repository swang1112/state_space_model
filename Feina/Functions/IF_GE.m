function output = IF_GE(draws)

tmp=sacf(draws,100,1);
% plot(tmp)

d = find(tmp<0.01,1);

%tmp=tmp(1:d);

output = 1+2*sum(tmp(1:d-1));

end

