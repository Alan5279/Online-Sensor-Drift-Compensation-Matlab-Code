%This function finds all the zero rows in a matrix
%Input:
%Matrix:= the matrix required for checking
%Output:
%RSet:= the index of rows where all values are zeros
%Number:= the number of all-zero rows in Matrix
%Coded by: Zhiyuan Ma
%Date: Sep. 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RSet,Number]=FindZeroRows(Matrix)
    [r,c]=size(Matrix);
    RSet=[];
    Number=0;
    flag=0;
    for i=1:r
        %Check for non zero values in a row
        for j=1:c
            if(Matrix(i,j)~=0)
                flag=1; %Change flag if there is non zero values
                break;  %Stop checking for the rest of the colunms
            end
        end
        %If flag remains 0, include the index and increase the Number
        if(flag==0)
            RSet=[RSet,i];
            Number=Number+1;
        end
        flag=0;
    end
end