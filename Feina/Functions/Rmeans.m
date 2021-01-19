% calculate recursive means
function Rmeans(store, nd, burn, str)
it = nd-burn; 

[T, N] = size(store); 
m = zeros(N, nd-burn); 
m(:, 1) = store(1, :)'; 

for j = 1:N
    for t = 2:nd-burn
        m(j, t) = (t-1)/t*m(j, t-1) + 1/t*store(t, j);
    end
end

RM = m'; 

[T, n]=size(RM);
q=numSubplots(n); % find parameters to optimally arrange subplots

figure;suptitle('Recursive Means of Retained Draws '); 

for j=1:n
    subplot(q(1,2),q(1,1),j)
    plot(RM(:,j),'linewidth' , 1.5);title(str(j))
end

end
