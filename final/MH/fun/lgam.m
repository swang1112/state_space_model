function Y=lgam(v,delta,X)
A=v/2;
B=2/delta;
Y = log(gampdf(X,A,B));