%This manuscript is used to implement RBF activation function with with
%setting
function [H]=SRBFun(P,IW,Bias,width)
ind=ones(size(P,1),1);
for i=1:size(IW,1)
    Weight=IW(i,:);         
    WeightMatrix=Weight(ind,:);
    V(:,i)=-sum((P-WeightMatrix).^2,2);    
end
BiasMatrix=Bias(ind,:);
V=V.*BiasMatrix;
H=exp(V);
end