%_________________________%
% Local Level Model: Nile %
%_________________________%

clear
clc
%%
Y = importdata("nile.txt");

[alpha, P, v, F] = local_level_filter(Y, 0, 10^7, 15099, 1469.1);


%%
subplot(2,2,1);
plot(2:100, [Y(2:end), alpha(2:end)], 'LineWidth', 1.5);
title("Data and alpha");
subplot(2,2,2);
plot(2:100, P(2:end));
title("state variance: P");
subplot(2,2,3);
plot(3:100, v(3:end));
title("prediction error: v");
subplot(2,2,4);
plot(3:100, F(3:end));
title("prediction error variance: f");

