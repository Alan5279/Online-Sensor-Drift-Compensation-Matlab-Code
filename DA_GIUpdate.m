%This function is used to form DAELM_T using generalized inverse solution
%instead of Lagrange method. 
%Description:
%       This function will be used to replace DAELM_TS in the Mainframe to
%       test its performance. The method used in this function is
%       generalized inverse solution. Different from Lagrange method, the
%       method in this function will not solve alpha_1, alpha_2, but
%       directly provide the solution based on ELM assumption. The method
%       takes IW, Bias and target domain to update a beta_T
%Input:
%       LData:= the labeled samples from target domain, consisting of
%       Labels and Features;
%       UnlData:= all the sample from target domain;
%       IW:= the randomized input weight which requires to be initialized
%       by RandomizeELM();
%       Bias:= the randomized input bias which requires to be initialized
%       by RandomizeELM();
%       ActType:= the activationfunction type;
%       Ct:= the regularization coefficient which is named Ct in the
%       formula;
%       Ctu:= the regularization coefficient which is named Ctu in the
%       formula;
%       BetaS:= the output weight matrix generated from source domain;
%Output:
%       BetaT:= the output weight matrix generated from target domain;
%Coded by: Zhiyuan Ma
%Date: Sep. 2017
%Status: tested
function [BetaT]=DA_GIUpdate(lData,UnlData,IW,Bias,ActType,Ct,Ctu,BetaS)
    [lr,lc]=size(lData);
%     [Unlr,Unlc]=size(UnlData);
    %Prepare the data
    TempTt=lData(:,1);
    Xt=lData(:,2:lc);
    Xtu=UnlData;
    clear lData UnlData;    %Free the memory
    
    NofClasses=6;
    %Format label
    Tt=zeros(lr,NofClasses)-1;
    for tempN=1:lr
        Tt(tempN,TempTt(tempN,1))=1;
    end %end of for 
    
    %Prepare the calculate the coresponding hidden layer output
    Ht=HOutput(Xt,IW,Bias,ActType);
    Htu=HOutput(Xtu,IW,Bias,ActType);
    NofHiddenNeurons=size(Ht,2);
    %Calculate BetaT
    if (lr>NofHiddenNeurons)  %Row > NofHiddenNeurons
        BetaT=inv(speye(NofHiddenNeurons)+Ct*(Ht'*Ht)+Ctu*(Htu'*Htu))*(Ct*Ht'*Tt+Ctu*(Htu'*Htu)*BetaS);
    else
        P=Ht*Ht';
        Q=Ht*Htu';
        BetaT=Ht'*pinv(speye(lr)+Ct*P+Ctu*pinv(P)*(Q*Q'))*(Ct*Tt+Ctu*pinv(P)*Q*Htu*BetaS);
    end%end if
end