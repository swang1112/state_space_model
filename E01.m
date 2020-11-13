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
% ich gehe davon das dass es mit matrix B gemeint ist.
A'
inv(A'*A)
det(A'*A)

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
z_sorted = z;
for i=2:100 
    for j=1:i
        if z_sorted(j) > z_sorted(i)
            foo = z_sorted(j);
            z_sorted(j) = z_sorted(i);
            z_sorted(i) = foo;
        end
    end
end
all(z_sorted == sort(z))

%% 6.
alpha = 0.1;
Ci = z_mean + z_std * tinv(1-alpha/2, 99) * [-1 ,1]
hist(z);

%% 7.
s = randn(100, 1);
s = sqrt(2)*s + 1;

%% 8.
X = randn(100, 1);
u = randn(100, 1);
beta = 5;
y = beta * X + u;
beta_hat = inv(X'*X) * X'* y
beta_hat - beta

% why is it not exactly 5? 
% Das ist nicht zu erwarten (beduetet auch nichts). 
% beta_hat wäre genau 5, wenn Fehlerterm deterministisch wäre, also u gleich
% 0 für jede Beobachtung.

%% 9.
Beta = zeros(1000,1);
for r=1:1000
    X = randn(100, 1);
    u = randn(100,1);
    y = beta * X + u;
    Beta(r) = inv(X'*X) * X'* y;
end
mean(Beta)
hist(Beta)
