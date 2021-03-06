README

1. mainfile
main.m:				main file (the other 2 mainfiles can be accessed in main.m) (--> produce out1.mat, out2.mat, out3.mat)
makedata.m: 		simulate data (--> produce true_par.mat, data.mat, R_i.mat, R_w.mat, h_i.mat, h_w.mat)
init_est.m:			maximize posterior and produce mode and hessian to init MH (--> produce mode.mat, post_val.mat, post_hess.mat)

2. function
kalman_garch_uni.m:	kalman filter
neg_posterior.m: 	calculate negative log posterior
getpar.m:			extract individual model parameters from a vector

3. datafile
true_par.mat:	true parameters
data.mat: 		observables 
R_i.mat:		country-specific components
R_w.mat: 		common components
h_i.mat:		conditional variances of country-specific shocks
h_w.mat:		conditional variances of common shocks
mode.mat:		posterior mode 
post_val.mat:	maximum of posterior (measured at estimated mode)
post_hess.mat: 	hessian matrix
out1.mat*:		drawn parameters
out2.mat*:		drawn states
out3.mat*:		drawn conditional volatilities

4. figure
par_hist.pdf:	histogram of drawn parameters
par_plot.pdf:	discriptive (trace-)plot of drawn parameters
states.pdf:		median of drawn states
cond_var.pdf: 	median of drawn conditional volatilities

* these datafiles are too big for email. They can be generated by running the 3.rd section of 'main.m' (line 7 - line 110)