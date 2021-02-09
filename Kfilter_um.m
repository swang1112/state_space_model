function [hStt,hPtt]=Kfilter_um(y,Z,X,S0,P0,Qt,Tt,K,H,mu)

[T,N] = size(y);
Ti = sum(isfinite(y));
k = size(X,2);
d = size(mu,2);
ns = size(S0,1);
hStt = zeros(ns,T);
hPtt = zeros(ns,ns,T);
Sti = S0; Pti = P0;

for t=1:T
    for i=1:N
        if t>T-Ti(1,i)
            if k==0
                V = y(t,i)-Z(i,:,t)*Sti;
            else
                V = y(t,i)-Z(i,:,t)*Sti-X(t,i);
            end        
            Fti = Z(i,:,t)*Pti*Z(i,:,t)'+ H(i,i,t);
            F_inv = inv(Fti);
            Mti = Pti*Z(i,:,t)';
            Sti = Sti + Mti*F_inv*V;
            Pti = Pti - Mti*F_inv*Mti';
        end
    end
    hStt(:,t) = Sti;
    hPtt(:,:,t) = Pti;
    if d==0
        Sti = Tt*Sti;
    else
        Sti = Tt*Sti + mu;
    end
    Pti = Tt*Pti*Tt'+K(:,:,t)*Qt*K(:,:,t)';
end