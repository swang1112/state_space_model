function out = dspectrum(x, omega)

% omega: desired frequency 
% kerbel: Bartlett
% bandwidth: 2*sqrt(T)
%see page 167 in Hamilton TSA

%kernel with bandwidth 2*sqrt(T)
T=size(x);
T=T(1);
q=2*sqrt(T);
xbar=mean(x);
j=0;
lamda0=sum((x(j+1:T)-xbar).*(x(j+1:T)-xbar))/T;

out=[];
for jj=1:length(omega)
    f=lamda0;
    f1=0;
    for j=1:q
        xlag=lag0(x,j);
        lamdaj=sum((x(j+1:T)-xbar).*(xlag(j+1:T)-xbar))/T;
        f1=f1+(1-(j/(q+1)))*lamdaj*cos(omega(jj)*j);
    end
    f=(f+2*f1)/(2*pi);
    out=[out;f];
end
