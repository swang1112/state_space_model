README

1. mainfiles 
main.m:			main file (the other 3 mainfiles can be accessed in main.m)
makedata.m: 	simulate data with parameters given in Table 2 Berger&Pozzi(2013)
kalman_garch.m:	kalman filter a la Harvey et al. 1992
MLE.m:			Maximum Likelihood Estimation

2. datafiles
data.mat: 		observables 
R_i.mat:		country-specific components
R_w.mat: 		common components
h_i.mat:		conditional variances of country-specific shocks
h_w.mat:		conditional variances of common shocks

3. miscellaneous
Kalman_with_true_par.m:		test kalman_garch.m with known parameters

