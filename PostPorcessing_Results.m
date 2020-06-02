%This manuscript is used for processing the results and plotting the
%figures.
% Folder='Data/With Error post back/DAELM-T/';
% Flag='LNum-';
% Flag='Time-';
Flag='Error-';
% Folder='Data/With Error post back/DAELMS/';

% Folder='Data/With Error post back/ELM/';

% Folder='Data/With Error post back/ODAELMS/';

% Folder='Data/With Error post back/ODAELMT/';
% Folder='Data/'
%%%%%%%%%%%%%%%%Same Sequence%%%%%%%%%%%%%%%%%%
% Folder='Data/SameSequence/ELM/';
% Folder='Data/SameSequence/SVM/';
% Folder='Data/SameSequence/Ensemble-ELM/';
% Folder='Data/SameSequence/Ensemble-SVM/';
% Folder='Data/SameSequence/RandomForest/';
% Folder='Data/SameSequence/DAELMS/';
% Folder='Data/SameSequence/DAELMT/';
% Folder='Data/SameSequence/ODAELMS/';
% Folder='Data/SameSequence/ODAELMT/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%Fixed Sequence%%%%%%%%%%%%%%%%%%
% Folder='Data/FixCircle/';
% Folder='Data/FixCircle/Add/';;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%Fixed Sequence%%%%%%%%%%%%%%%%%%
% Folder='Data/IntegratedExp/';
Folder='Data/NoError/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%Methods%%%%%%%%%%%%%%%%%%%%%%%%%
% FileName='DAELMS-';
% FileName='DAELMT-';
% FileName='ODAELMS-';
% FileName='ODAELMT-';
% FileName='Ensemble-ELM-';
% FileName='Ensemble-SVM-';
% FileName='RandomForest-';
% FileName='ELM-';
FileName='SVM-';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%Batch No.%%%%%%%%%%%%%%%%%%%%%%%
Index='B';
% Index='B1';
% Index='B2';
% Index='B3';
% Index='B4';
% Index='B5';
% Index='B6';
% Index='B7';
% Index='B8';
% Index='B9';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%Setting No. %%%%%%%%%%%%%%%%%%%%%%%%%
% SameName='Same-';
SameName='';
% FixName='Fix-';
% FixName='Fix2-';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Folder='Data/';
% Folder='Data/FixCircle/';

FileType='.txt';
%%%%%%%%%%%%%%%%
File=[Folder,FileName,Flag,Index,FileType];
A=zeros(9,1);
for i=1:9
    File=[Folder,FileName,SameName,Flag,Index,num2str(i),FileType];
    DA=load(File);
%     TempE=sum(DA);
%     TempE=(1-mean(DA))*100;
TempE=(1-DA(end,1))*100;
    A(i,1)=TempE;
end
A
mean(A)

% File=[Folder,FileName,Flag,Index,FileType];
% DA=load(File);
% figure;
% plot(DA);
% set(gca,'FontSize',35);
% The followings are for processing time
% Folder='Data/With Error post back/DAELM-T/';
% FileName='DAELMT-Time-';
% Index='B1';
% FileType='.txt';
% File=[Folder,FileName,Index,FileType];
% DA=load(File);
% 
% Time=sum(DA);
% %[,DAELMS]
% BarTime=[67.8,36.4,]
% 
% figure;
% bar(BarTime);