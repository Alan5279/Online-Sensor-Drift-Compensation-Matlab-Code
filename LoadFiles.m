%This function is used for loading data from DataFiles.
%Output:a cell matrix that has all the data
%       Format: [Target,Features];
%The file folder and the file names are built in.
function [Data]=LoadFiles()
    %Initialize a cell matrix as Data
    Data=cell(1,10);
    %Data Folder
    MainFolder='Data';

    Path=[MainFolder];

    %DataRegExp='^batch[0-9]{1,2}.dat$';
    DataRegExp='batch[1-9].dat$';
    DataRegExp2='batch([1-9])+0.dat';

    %Load all the file names into matlab
    datafileFolder=fullfile(Path);
    dirOutput=dir(fullfile(datafileFolder,'*'));
    rawDataFileNames={dirOutput.name}';

    %Find the data file
    DataFileNames=cell(10,1);
    NumFlag=1;

    %In order to seperate 10 from 1-9, use two for loop
    for k=1:size(rawDataFileNames,1)
        if(~isempty(regexp(rawDataFileNames{k,1},DataRegExp,'match')))
            DataFileNames{NumFlag,1}=rawDataFileNames(k,1);
            NumFlag=NumFlag+1;
        else
            continue;
        end
    end
    for k=1:size(rawDataFileNames,1)
        if(~isempty(regexp(rawDataFileNames{k,1},DataRegExp2,'match')))
            DataFileNames{NumFlag,1}=rawDataFileNames(k,1);
            NumFlag=NumFlag+1;
        else
            continue;
        end
    end

    %Load Data. This part of code implements loading data using one character
    %by one character
    % for i=1:1
    %     Temp=load([Path,filesep,char(DataFileNames{i,1})]);
    %     fid=fopen([Path,filesep,char(DataFileNames{i,1})],'r');
    %     temp=fread(fid,'*char');
    %     disp(temp);
    %     Temp='';
    %     for j=1:size(temp,1)
    %         if (strcmp(temp(j,1),','))
    %             Temp=[Temp,' '];
    %         elseif(strcmp(temp(j,1),':'))
    %             tail=size(Temp,2);
    %             while(~strcmp(Temp(1,tail),' '))
    %                 Temp=Temp(1,1:size(Temp,2)-1);
    %                 tail=tail-1;
    %                 if(tail==0)
    %                     continue;
    %                 end
    %             end
    %             
    %         elseif(strcmp(temp(j,1),';'))
    %             Temp=[Temp,' '];
    %         else
    %             Temp=[Temp,temp(j,1)];
    %         end
    %     end
    % end

    Iexp='';
    for k=1:129
        if(k==1)
            Iexp='%f;%f';
        else
            Iexp=[Iexp,'%*f:%f'];
        end

    end

    for i=1:10
        %Temp=load([Path,filesep,char(DataFileNames{i,1})]);
        fid=fopen([Path,filesep,char(DataFileNames{i,1})]);

        Temp=textscan(fid,Iexp,'delimiter',' ');
        Test=cell2mat(Temp);
        [pn,ps] = mapminmax(Test(:,3:end)',0,1);    %original paper omit the field of concentration
%         [tn,ts] = mapminmax(Data(:,1)',-1,1);
        
        %Normalize the data
        
        Data{i}=[Test(:,1),pn'];
    end
end