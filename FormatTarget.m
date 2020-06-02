%This function is used to format the output such that only one dimension is
%Pos and the rest are Neg
%Description: 
%       This function is used for multi-classification problem. The target
%       should be formatted into [...,Neg,Pos,Neg,...] where the index of
%       Pos represent the class label. The value Neg and Pos represent the
%       non-class label and class label. The function chooses the largest
%       value in source (i,:) and the index of the value is the class
%       label.
%Output:
%       Target:=formatted target vector n*m where n is the number of
%       samples and m is the number of classes
%Coded by: Zhiyuan Ma
%Date: Sep. 2017
function [Target]=FormatTarget(Source,Pos,Neg)
    [r,c]=size(Source);
    Target=zeros(size(Source))+Neg; %Set all the values to Neg
    for i=1:r
        %Locate the maximum value
        [MaxV,Index]=max(Source(i,:));
        
        %Set the proper format value for target
        for j=1:c
            if j==Index
                Target(i,j)=Pos;
            else
                continue;
            end
        end%end of for j
        
    end%end of for i
end%end of function