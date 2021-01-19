% plot Autocorrelation
function auto(TR,str)


[T, n]=size(TR);
q=numSubplots(n); % find parameters to optimally arrange subplots

figure; suptitle('Autocorrelation of Retained Draws'); 

for j=1:n
    subplot(q(1,2),q(1,1),j)
    plot(autocorr(TR(:,j)), 'r--.', 'linewidth' ,1);title(str(j)); 
end

end
