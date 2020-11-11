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
transpose(A);
inv(A);
det(A)