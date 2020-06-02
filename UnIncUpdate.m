%This manuscript is used to implement the function of unlabeled incremental
%learning.
%Input:
%
%Coded by: Zhiyuan Ma
%Date: Oct. 2017.
%Status: untested
function [beta,K]=UnIncUpdate(beta0,K0,h,HT,betaS,C_Tu,Case)
    if(Case==1)
        %Case 1 where HT has no less rows than columns;
        Temp=C_Tu*h*K0*h';
        K=K0-C_Tu*K0*h'*pinv(eye(size(Temp))+Temp)*h*K0;
        beta=beta0-K0*C_Tu*(h'*h)*(beta0-betaS);
        clear Temp;
    elseif(Case==2)
        %Case 2 where HT has more columns than rows
        Tempk=h*HT';
        IP=pinv(HT*HT');
        Temp=Tempk'*Tempk*K0*C_Tu*IP;
        K=K0-K0*C_Tu*IP*pinv(eye(size(Temp))+Temp)*(Tempk'*Tempk)*K0;
        beta=beta0-C_Tu*HT'*K*IP*HT*(h'*h)*(beta0-betaS);
    else
        fprintf('The parameter case in UnDecUpdate is invalid!\t\n');
    end%end if elseif else
end