%This manuscript is used to implement the incremental approach in ODAELM-S.
%The function is used for MainFrame_ODAELMS.m and other related files.
%Input:
%   beta0:= the output weight matrix of ELM
%   K0:= the intermediate result 
%   PI:=pinv(HS*HS'); for case 2 only
%   h:=the increment of the labele sample(s)
%   HS:=the hidden layer output of the source domain
%   t:= the increment of the labele sample(s)
%   C_T:=the regularization parameter
%   Case:=flag for updates
%Output:
%   beta:= the updated output weight matrix
%   K:= the updated intermediate result K^{-1}
%Coded by: Zhiyuan Ma
%Date: Nov. 2017
%Status: tested
function [beta,K]=SourceIncUpdate(beta0,K0,PI,h,HS,t,C_T,Case)
    if(Case==1)
        %Case 1 where HS has no less rows than columns;
        Temp=C_T*h*K0*h';
        K=K0-C_T*K0*h'*pinv(eye(size(Temp))+Temp)*h*K0;
        beta=beta0-C_T*K*h'*(h*beta0-t);
        clear Temp;
    elseif(Case==2)
        %Case 2 where HS has more columns than rows
        Tempk=h*HS';
        Tempktk=Tempk'*Tempk;
        Temp=C_T*(Tempktk)*K0*PI;
        K=K0-K0*C_T*PI*pinv(eye(size(Temp))+Temp)*(Tempktk)*K0;
        beta=beta0-C_T*HS'*K*PI*Tempk'*(h*beta0-t);
        clear Tempk Temp;
    else
        fprintf('The parameter case in UnDecUpdate is invalid!\t\n');
    end%end if elseif else
end