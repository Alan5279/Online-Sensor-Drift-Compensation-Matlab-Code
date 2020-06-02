%This manuscrip is used for Ensemble_SVM
%This manuscript is used for implement Ensemble SVM
%Initialize results
Earray=cell(length(TestingCell));
LNumberarray=cell(length(TestingCell));
%Begin training and testing
MSet=cell(10,1); %initialize the cell array for storing SVM model
BSet=zeros(10,1);
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    [LSSet,TempSet]=kenstone(TestSet(:,2:endsize),50);%LSet is the labeled set 
    
    if i==1
        Xs=TrainSet(:,2:endsize);
        Ts=TrainSet(:,1);
        model=svmtrain(Ts,Xs,'-s 0 -t 2 -c 2 -g 0.02');   %2 is the rbf kernel
        MSet{i}=model;
        [PT,accuracy,decision_values]= svmpredict(Ts,Xs,model); %Testing
        BSet(i,1)=accuracy(1,1);
    else
        Xs=TrainSet(:,2:endsize);
        Ts=TrainSet(:,1);
        model=svmtrain(Ts,Xs,'-s 0 -t 2 -c 2 -g 0.02');   %2 is the rbf kernel
        MSet{i}=model;
        [PT,accuracy,decision_values]= svmpredict(Ts,Xs,model); %Testing
        BSet(i,1)=accuracy(1,1);
    end
    
    
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
        X=TestSet(1:j,2:endsize);   %Get all the features
        T=TestSet(1:j,1);
        FPT=zeros(size(T,1),6);
        for TempIndex=1:i-1;
            %Extract model
            TempM=MSet{TempIndex};
            %Calculate output
            [PT,accuracy,decision_values]= svmpredict(T,X,TempM); %Testing
            TT=zeros(size(PT,1),6)-1;
            for Tempk=1:size(PT,1)
                TT(Tempk,PT(Tempk,1))=1;
            end
            FPT=FPT+TT*BSet(TempIndex,1)/100;
        end
        %Calculate the residual error
        FPT=FormatTarget(FPT,1,-1);
        FT=zeros(size(T,1),6)-1;
        for Tempk=1:size(T,1)
            FT(Tempk,T(Tempk,1))=1;
        end
        [RSet_sourceP,No_sourceP]=FindZeroRows(FPT-FT);
        NofSamples=size(FT,1);
        Error=1-No_sourceP/NofSamples;%Calculate the recall rate for each method
        ESet(j,1)=Error;
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['Ensemble-SVM-Error-B',num2str(i)];
    LNumSubFolder=['Ensemble-SVM-LNum-B',num2str(i)];
    TimeSubFolder=['Ensemble-SVM-Time-B',num2str(i)];
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