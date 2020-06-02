%This manuscript is used to processing the raw data and load them into
%files
function [TPDomain,TTDomain]=LoadData(MainFolder)

%Data Folder
%MainFolder='Data';

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
%
TestFlag=1;
%Initialize the source domain and target domain
TPDomain=cell(10,1);    %Take the next 9 batches as target domain. Target domain pattern
TTDomain=cell(10,1);   %Target domain targets
for i=1:10
    %Temp=load([Path,filesep,char(DataFileNames{i,1})]);
    fid=fopen([Path,filesep,char(DataFileNames{i,1})]);
    
    Temp=textscan(fid,Iexp,'delimiter',' ');
    Test=cell2mat(Temp);
    
    %Preparing data for learning
    Dimension=size(Test,2);
    if(i==1)
        %P0=Test(:,3:Dimension);
%         TempP=Test(:,3:Dimension);
TempP=Test(:,2:Dimension);
%         P0=mapminmax(TempP',-1,1);  %Normalized pattern
%         P0=P0';
P0=NormData(TempP);
        clear TempP;
        T0=Test(:,1);
%         TempP=Test(:,3:Dimension);
TempP=Test(:,2:Dimension);
%         TempR=mapminmax(TempP',-1,1);
%         TPDomain{i,1}=TempR';
        TempR=NormData(TempP);
        TPDomain{i,1}=TempR;
        clear TempP;
        clear TempR;
        TTDomain{i,1}=Test(:,1);
    else
        %Preparing target domains
%         TPDomain{i}=Test(:,3:Dimension);
%         TempP=Test(:,3:Dimension);
TempP=Test(:,2:Dimension);
%         TempR=mapminmax(TempP',-1,1);
%         TPDomain{i,1}=TempR';
        TempR=NormData(TempP);
        TPDomain{i,1}=TempR;
        clear TempP;
        clear TempR;
        TTDomain{i,1}=Test(:,1);
    end
    %The two rows below are for testing
%     if(i==TestFlag)
% %     TestP=mapminmax(Test(:,3:Dimension));
% TempP=Test(:,2:Dimension);
%     TestT=Test(:,1);
%     end
end
end

function [NormData]=NormData(P)
    NormData=zeros(size(P));
    NormData=P;
    for i=1:size(P,2)
        NormData(:,i)=P(:,i)/max(abs(P(:,i)));
    end
end
