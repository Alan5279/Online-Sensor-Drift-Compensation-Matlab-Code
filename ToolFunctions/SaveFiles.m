%This function is used to save a matrix into a destination using comma as
%its separate
%Input:
%   DataMatrix:= the matrix containing data
%   FileNames:=the path and the file name for storing
%Description:
%   The matrix is saved in a format where each field is separated by comma
%   and each row is ended with semicolon;
%Coded by: Zhiyuan Ma
%Date: Oct. 2017
function SaveFiles(DataMatrix,FileNames)
    %Save files
    fid=fopen(FileNames,'w');
    [r,c]=size(ResultMatrix);
    for ri=1:r
        for ci=1:c
            if ci==c
    %             fprintf(fid,'%3f\r\n',ResultMatrix(ri,ci));
                fprintf(fid,'%3f;\r\n',DataMatrix(ri,ci));
            else
    %             fprintf(fid,'%3f\t',ResultMatrix(ri,ci));
                fprintf(fid,'%3f,',DataMatrix(ri,ci));
            end
        end
    end
    fclose(fid);
end