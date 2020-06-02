%This manuscript is used to implement the algorithm of DAELM-T
%This realization is highly unreliable
function [beta]=DAELMT(betas,IW,Bias,LPattern,LTarget,ULPattern,Ct,Ctu,ActType)
    %samples
    [Ht]=HOutput(LPattern,IW,Bias,ActType);
    [Htu]=HOutput(ULPattern,IW,Bias,ActType);
    Tt=LTarget;
    
    %Thirdly, update the output weight matrix
    [r,c]=size(Ht);
    if(r>=c)
        %Case1
        I=eye(c);
%         beta=pinv(I+Ct*(Ht'*Ht)+Ctu*(Htu'*Htu)+Cts*(Ht'*Ht))*(Ct*(Ht'*Ht) + Ctu*(Htu'*Htu)*betas + Cts*(Ht'*Ht)*betas);
%         beta=pinv(I+ C*(Ht'*Ht)+ Ctu*(Htu'*Htu))*(Ct*Ht'*Tt + Ctu*(Htu'*Htu)*betas);
        beta=(I+ C*(Ht'*Ht)+ Ctu*(Htu'*Htu))\(Ct*Ht'*Tt + Ctu*(Htu'*Htu)*betas);
%         beta=inv(I+ C*(Ht'*Ht)+ Ctu*(Htu'*Htu))*(Ct*Ht'*Tt +
%         Ctu*(Htu'*Htu)*betas);   %Less satistfying result
    else
        %Case2
        It=eye(r);
        [rtu,ctu]=size(Htu);
        Itu=eye(rtu);
        TempP=Htu*Htu'+Itu/Ctu;   %r*r
        TempOmega=Ht*Htu';  %r*rtu
        TempR=Ht*Ht'+It/Ct;
        TempO=Htu*Ht';
        Ttu=Htu*betas;
        Alphat=inv(TempOmega*inv(TempP)*TempO-TempR)*(TempOmega*inv(TempP)*Ttu-Tt);
        Alphatu=inv(TempP)*Ttu-inv(TempP)*TempO*inv(TempOmega*inv(TempP)*TempO-TempR)*(TempOmega*inv(TempP)*Ttu-Tt);
        betatest=Ht'*Alphat + Htu'*Alphatu;
        beta=betatest;
%         beta=Ht'*pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*Htu*betas-Tt)+Htu'*(pinv(TempP)*Htu*betas - pinv(TempP)*TempO*pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*Htu*betas-Tt));
        
%         Ans=beta-betatest
%         beta=Ht'*(TempOmega*TempP^(-1)*TempO-TempR)^(-1)*(TempOmega*TempP^(-1)*Htu*betas-Tt)+Htu'*(TempP^(-1)*Htu*betas - TempP^(-1)*TempO*(TempOmega*TempP^(-1)*TempO-TempR)*(TempOmega*TempP^(-1)*Htu*betas-Tt));
    end
end