%_____%
% E01 %
%_____%

clc
clear
%% 1.
B = randn(3,4);
B(1,1) = 0;
B(:,2) = [];

%% 2.
A = randn(2,5);
D = A(:,1:3);
F = A(:,[2,4]);

%% 3.
B'
inv(B)
det(B)

%% 4.
z = randn(100,1);
z_mean = sum(z) / 100;
mean(z) - z_mean
square_div = z-z_mean;
z_var = square_div' * square_div /99;
z_std = sqrt(z_var);
z_var - var(z)
z_std - std(z)

%% 5.
clc
for i=1:100
    if
end
