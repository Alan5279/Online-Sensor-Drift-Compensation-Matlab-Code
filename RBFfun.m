%P:=n_s*m matrix, where m is the feature dimension number
%Center:=IW
function [H]=RBFfun(Pattern,Center,Width)
    Norm=zeros(size(Pattern,1),size(Center,1));
    ind=ones(Pattern,1)
    for i=1:size(Center,1)
        Weight
        TempNorm=Pattern-Center(ind,:);
        Norm(:,i)=sum(TempNorm.^2)/Width(1,1);      
    end
    H=exp(-Norm);
    
end