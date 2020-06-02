%This manuscript is used for Ensemble_ELM_Same
%Status:tested
%Initialize results
Earray=cell(length(TestingCell));
LNumberarray=cell(length(TestingCell));
%Begin training and testing
ModelSet=cell(10,1); %For storing ELM paramters
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
        %Initialize the base ELM as source domain classifier
        TrainingData_File=[Ts,Xs];
        TestingData_File=[Ts,Xs];
        T=TrainingData_File(:,1);
        Elm_Type=1;
        C=0.001;
        %Set parameters
        if (length(Ts)>nHiddenNeurons)
            NL=1;
        else
            NL=0;
        end
        [IW,Bias,BetaS] = ELM_S(TrainingData_File, TestingData_File, Elm_Type, nHiddenNeurons, ActType,C,NL);
        [Hs]=HOutput(Xs,IW,Bias',ActType);
        %Caclulate the output weight matrix beta
        Format_Ts=zeros(size(Ts,1),NofClasses)-1;
        for tempN=1:size(Format_Ts,1)
            Format_Ts(tempN,Ts(tempN,1))=1;
        end %end of for 
            
        %Calculate the recall rate
        T_sourceP=Hs*BetaS;
        Target_sourceP=FormatTarget(T_sourceP,1,-1);
        [RSet_sourceP,No_sourceP]=FindZeroRows(Target_sourceP-Format_Ts); %Calculate the proper classified samples
        NofSamples=size(Format_Ts,1);
        BSet(i,1)=No_sourceP/NofSamples*100;%Calculate the recall rate for each method
        Model={IW,Bias,BetaS};
        ModelSet{i}=Model;
    else
        Xs=TrainSet(:,2:endsize);
        Ts=TrainSet(:,1);
        %Update source domain.
        %Train a base classiffier on current domain
        %Set patterns and targets
        %Initialize the base ELM as source domain classifier
        TrainingData_File=[Ts,Xs];
        TestingData_File=[Ts,Xs];
        T=TrainingData_File(:,1);
        Elm_Type=1;
        C=0.001;

        %Set parameters
        if (length(Ts)>nHiddenNeurons) 
            NL=1;
        else
            NL=0;
        end

        [IW,Bias,BetaS] = ELM_S(TrainingData_File, TestingData_File, Elm_Type, nHiddenNeurons, ActType,C,NL);
        [Hs]=HOutput(Xs,IW,Bias',ActType);
        %Caclulate the output weight matrix beta
        Format_Ts=zeros(size(Ts,1),NofClasses)-1;
        for tempN=1:size(Format_Ts,1)
            Format_Ts(tempN,Ts(tempN,1))=1;
        end %end of for 

        %Calculate the recall rate
        T_sourceP=Hs*BetaS;
        Target_sourceP=FormatTarget(T_sourceP,1,-1);
        [RSet_sourceP,No_sourceP]=FindZeroRows(Target_sourceP-Format_Ts); %Calculate the proper classified samples
        NofSamples=size(Format_Ts,1);            
        BSet(i,1)=No_sourceP/NofSamples*100; %Store the accuracy as weight for ensembling
        Model={IW,Bias,BetaS};
        ModelSet{i}=Model;
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
        X_te=TestSet(1:j,2:endsize);   %Get all the features
        TestT=TestSet(1:j,1);
        for t=1:i;
            IW=ModelSet{t}{1};
            Bias=ModelSet{t}{2};
            BetaS=ModelSet{t}{3};
            weight=BSet(t,1);
            [H_te]=HOutput(X_te,IW,Bias',ActType);
            TempT=H_te*BetaS;
            if(t==1)
                T_P1=TempT*weight;
            else
                T_P1=T_P1+TempT*weight;
            end
        end
        %Format target
        Format_TestT=zeros(size(TestT,1),NofClasses)-1;
        for tempN=1:size(Format_TestT,1)
            Format_TestT(tempN,TestT(tempN,1))=1;
        end %end of for 
        Target_P1=FormatTarget(T_P1,1,-1);
        %Calculate the accuracy
        [RSet_sourceP,No_P1]=FindZeroRows(Target_P1-Format_TestT); %Calculate the proper classified samples
        NofSamples=size(Format_TestT,1);
        Error=1-No_P1/NofSamples;%Calculate the recall rate for each method
        ESet(j,1)=Error;
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['Ensemble-ELM-Same-Error-B',num2str(i)];
    LNumSubFolder=['Ensemble-ELM-Same-LNum-B',num2str(i)];
    TimeSubFolder=['Ensemble-ELM-Same-Time-B',num2str(i)];
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