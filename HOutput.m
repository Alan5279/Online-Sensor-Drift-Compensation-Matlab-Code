%This manuscript is used to calculate the hidden layer output of ELM
function [H]=HOutput(Pattern,IW,Bias,ActType)
    switch lower(ActType)
        case{'rbf'}
%             H = RBFun(Pattern,IW,Bias);
            %The following are using radbas function to generate RBF output
            V=Pattern*IW'; ind=ones(1,size(Pattern,1));
            BiasMatrix=Bias(ind,:);      
            V=V+BiasMatrix;
            H=radbas(V);
        case{'sig'}
            H = SigActFun(Pattern,IW,Bias);
        case{'sin'}
            H = SinActFun(Pattern,IW,Bias);
        case{'hardlim'}
            H0 = HardlimActFun(Pattern,IW,Bias);
            H = double(H0);
    end
end