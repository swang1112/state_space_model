function [out] = flatprior(x,c,LB,UB)
% flat prior
if x > LB 
    if x < UB
        out = c;
    else
        out = 0;
    end
else
    out = 0;
end 
end
