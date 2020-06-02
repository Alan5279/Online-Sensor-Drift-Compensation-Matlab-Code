%This manuscript is used to implement the update of unlabeled decrement
%Input:
%   beta0:= the output weight before udpate;
%   K0:= the intermediate result before update;
%   HT:= the hidden layer output of the labeled samples before update;
%   h:= the incremental row of HT;
%   C_Tu:= the coefficient C_Tu in DAELM;
%   betaS:= the output weight of base classifier;
%   Case:= the parameter for determining the case of the calculation, 1 is
%   for case 1 and 2 for case 2
%Output:
%   beta:= the updated output
%   K:= the inverse of the updated intermediate result.
%Coded by: Zhiyuan Ma
%Date: Oct. 2017
%Status: untested.
function [beta,K]=UnDecUpdate(beta0,K0,HT,h,C_Tu,betaS,Case)
    %Judge the case
    if(Case==1)
        %Case 1 where HT has no less rows than columns
        Temp=C_Tu*h*K0*h';     
        K=K0+C_Tu*K0*h'*pinv(eye(size(Temp))-Temp)*h*K0;
        beta=beta0+C_Tu*(h'*h)*(beta0-betaS);
        clear Temp;        
    elseif(Case==2)
        %Case 2 where HT has more columns than rows
        Tempk=h*HT';
        IP=pinv(HT*HT');
        Temp=C_Tu*(Tempk'*Tempk)*K0*IP;
        K=K0+C_Tu*K0*IP*pinv(eye(size(Temp))-Temp)*(Tempk'*Tempk)*K0;
        beta=beta0+C_Tu*HT'*K*IP*HT*(h'*h)*(beta0-betaS);
        clear Temp;
    else
        fprintf('The parameter case in UnDecUpdate is invalid!\t\n');
    end
end