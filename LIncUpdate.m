%This manuscript is used to implement the function of labeled incremental
%laerning
%Input:
%   beta0:= the output weight before udpate;
%   K0:= the intermediate result before update;
%   C_T:= the coefficient C_T in DAELM;
%   C_Tu:= the coefficient C_Tu in DAELM;
%   h:= the incremental row of HT;
%   t:= the corresponding target of h;
%   HT:= the hidden layer output of the labeled samples before update;
%   HTu:= the hidden layer output of the unlabeled samples;
%   Case:= the flag parameter, 1 for case 1 and 2 for case 2. Others are
%   invalid.
%Output:
%   beta:= the output weight matrix after update;
%   K:= the inverse of updated intermediate result.
%Coded by: Zhiyuan Ma
%Date: Oct. 2017
%Status: untested
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [beta,K]=LIncUpdate(beta0,K0,C_T,C_Tu,h,t,HT,HTu,Case)
    %Judge the case
    if(Case==1)
        %Case 1 where HT has no less rows than columns
        Temp=C_T*h*K0*h';
        K=K0-C_T*K0*h'*pinv(eye(size(Temp))+Temp)*h*K0;
        beta=beta0-C_T*K*h*(h*beta0-t);
        clear Temp;
    elseif(Case==2)
        %Case 2 where HT has more columns than rows
        %Caclulate the partitioned part of K
        B=C_T*HT*h'+C_Tu*pinv(HT*HT')*HT*(HTu'*HTu)*h';
        C=C_T*h*HT';
        Temp=h*h';
        D=eye(size(Temp))+C_T*Temp;
        clear Temp;

        %Calculate the inverse of K
        F20=pinv(D-C*K0*B);
        Temp1=B*F20*C*K0;
        K11=K0*(eye(size(Temp1))+Temp1);
        K12=-K0*B*F20;
        K21=-F20*C*K0;
        K22=F20;
        K=[K11, K12 ; K21, K22];
        clear Temp1;

        %Calculate beta
        Temp=h*beta0-t;
        %     beta=beta0+HT'*K0*B*F20*C_T*Temp-C_T*h'*F20*Temp;
        beta=beta0+HT'*(-K12)*C_T*Temp -C_T*h'*F20*Temp;
        
%         %The following are modified version
%         TempHT=[HT;h];
%         TempTu=HTu'*HTu;
%         TempQQ=[HT*(TempTu)*HT',HT*TempTu*h;h*TempTu*HT',h*TempTu*h'];
%         TempP=[HT*HT',HT*h';h*HT',h*h'];
%         K=pinv(eye(size(TempP))+C_T*TempP+C_Tu*pinv(TempP)*TempQQ);
%         TempT=[HT*K0*beta0;t];
%         
%         beta=TempHT*K*(C_T*TempT+C_Tu*pinv(TempP)*TempHT*(HTu'*HTu)*betaS);
    else
        fprintf('The parameter case in UnDecUpdate is invalid!\t\n');
    end %end if else if

end%end function