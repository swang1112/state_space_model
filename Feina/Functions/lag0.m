function out=lag0(x,p)                                                                   
%lag the x p periods and places zeros in the before that 
[R,C]=size(x);
x1=x(1:(R-p),:);
out=[zeros(p,C); x1];
end