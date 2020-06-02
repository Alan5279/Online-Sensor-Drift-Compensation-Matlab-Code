%This manuscript is used to implement traditional SVM
%This manuscript is for IntegratedExp and the algorithm is SVM
%Status:tested
% TrainingCell=load('Data/RandomD/TraininngCell.mat');
% TestingCell=load('Data/RandomD/TestingCell.mat');
% TrainingCell=struct2cell(TrainingCell);
% TrainingCell=TrainingCell{1};
% TestingCell=struct2cell(TestingCell);
% TestingCell=TestingCell{1};
% K_Type='rbf';
% C=0.5;  %Penalty factor?
% %Initialize the parameters
% nHiddenNeurons=1000;
% ActType='rbf';
% Cs=0.1;
% Ct=100;
% Ctu=Cs;
% NofClasses=6; %The number of classes in the table
% endsize=size(TrainingCell{1},2);    %Set the end size
% %Initialize results
% Earray=cell(length(TestingCell));
% LNumberarray=cell(length(TestingCell));
%Begin training and testing
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    %Randomize TestSet
    
    Xs=TrainSet(:,2:endsize);
    Ts=TrainSet(:,1);
    model=svmtrain(Ts,Xs,'-s 0 -t 2 ');   %2 is the rbf kernel-c 2 -g 0.02
    clear Elm_Type NL; %Release the memory
    
    %Begin Initialize and update target ELM.Starting from 0;
    %In this case, the target ELM  starts with 0 unlabeled data and 1
    %labeled sample.
    %Initialize the parameters used for update
    x=0;y=0;    %x:=the number of total samples; y:= the number of labeled samples
    LSet=[];USet=[];    %LSet:= the set of labeled samples
    for j=1:length(TestSet)
        fprintf('This is the %d sample.\r\n',j);
        tic;
        %Firstly, calculate the current residual classification error rate
        x=j;    %Set x to the number of samples received so far
        %Calculate the residual error
        Xs=TestSet(1:j,2:endsize);   %Get all the features
        Ts=TestSet(1:j,1);
        [PT,accuracy,decision_values]= svmpredict(Ts,Xs,model); %Testing
        Error=1-accuracy(1,1)/100;
        ESet(j,1)=Error;
        
        clear TempT RSet Number;
        
        %Secondly, calculate the probabilities for labeling
        TempP=PSSA(x,y,Error);
%         TempP=PSSA(0,0,Error);  %Update based on fixed probability
        %Judge if labeling process is required
        if (rand(1,1)>TempP||Error<0.1)%No KS selection process
            %Do nothing
        else%Labeling process required
            %Find labeling samples
            if (x<=2)%There is only one or two samples so far
                %Do nothing
            else
                %There is more than 3 samples samples.
                %Choose the one 
                if (y==0)
                    %Choose the first two of KS algorithm
                    TempSet=TestSet(1:j,2:endsize);
                    Selected= 2;
                    [model,test]=kenstone(TempSet,Selected);
                    IncSet=model;
                    LSet=[model];
                    y=2;    %Set the labele sample to 2;
                    clear TempSet;
                    
                    %This is the first time for labeling, therefore the
                    %first time for initialize betaS new as well
                    dx=TestSet(IncSet,2:endsize);
                    
                    Xs=[Xs;dx];
                    Ts=[Ts;TestSet(LSet,1)];
                    model=svmtrain(Ts,Xs,'-s 0 -t 2 ');   %2 is the rbf kernel-c 2 -g 0.02
                else
                    %Choose the farthest nearest samples.
                    TempSet=TestSet(1:j,2:endsize);
                    Selected= y+1;
                    [model,test]=kenstone(TempSet,Selected);
                    IncSet=setdiff(model,LSet); %Get the samples that are not included in the IncSet
                    IncSet=IncSet(1);%So far, let us first increase only
%                     the first one; Now, test on all IncSet
                    LSet=[LSet,IncSet];  
                    y=y+length(IncSet);
                    clear TempSet;      
                    
                    %Doing incremental learning on ODAELM-S    
%                     dx=TestSet(IncSet,2:endsize);
%                     Tempt=TestSet(IncSet,1);
%                     %Format t
%                     t=zeros(size(Tempt,1),NofClasses)-1;
%                     for tempN=1:size(t,1)
%                         t(tempN,Tempt(tempN,1))=1;
%                     end
%                     h=HOutput(dx,IW,Bias',ActType);
                    Xs=[Xs;TestSet(LSet,2:endsize)];
                    Tempt=TestSet(LSet,1);
                    Ts=[Ts;Tempt];
                    model=svmtrain(Ts,Xs,'-t 2');   %2 is the rbf kernel
                    
                end%end y==0
            end %end x<2
        end %rand(1,1)<=TempP
        clear TempP;
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['SVM-Error-B',num2str(i)];
    LNumSubFolder=['SVM-LNum-B',num2str(i)];
    TimeSubFolder=['SVM-Time-B',num2str(i)];
    ErrorSaveNames=['Data',filesep,ErrorSubFolder,'.txt'];
    LNumSaveNames=['Data',filesep,LNumSubFolder,'.txt'];
    TimeSaveNames=['Data',filesep,TimeSubFolder,'.txt'];
    %Save the errors
    fid=fopen(ErrorSaveNames,'w');
    [r,c]=size(ESet);
    for ri=1:r
        fprintf(fid,'%3f;\r\n',ESet(ri,1));
    end
    fclose(fid);
    
    %Save the LNumber
    fid=fopen(LNumSaveNames,'w');
    [r,c]=size(LNumSet);
    for ri=1:r
        fprintf(fid,'%3f;\r\n',LNumSet(ri,1));
    end
    fclose(fid);
    
    %Save the time
    fid=fopen(TimeSaveNames,'w');
    [r,c]=size(TimeSet);
    for ri=1:r
        fprintf(fid,'%3f;\r\n',TimeSet(ri,1));
    end
    fclose(fid);
end