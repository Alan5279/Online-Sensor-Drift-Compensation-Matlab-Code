%This manuscript is used to implement traditional ELM-Kernal
%This manuscript is for IntegratedExp and the algorithm is ELM
%Status:tested
% TrainingCell=load('Data/RandomD/TrainingCell.mat');
% TestingCell=load('Data/RandomD/TrainingCell.mat');
%Begin training and testing
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    [LSSet,TempSet]=kenstone(TestSet(:,2:endsize),50);%LSet is the labeled set 
    %Initialize base ELM network.
    TempTs=TrainSet(:,1);
    %Format Ts
    Ts=zeros(size(TempTs,1),NofClasses)-1;
    for tempN=1:size(Ts,1)
        Ts(tempN,TempTs(tempN,1))=1;
    end
    clear TempTs;
    if (length(Ts)>nHiddenNeurons)
        NL=1;
        Case=1;%Flag for incremental learning
    else
        NL=0;
        Case=2;%Flag for inc learning
    end%end if NL
    Elm_Type=1;
    %If this is the first batch, initialize the base classifier
    [IW,Bias,betaS] = ELM_S(TrainSet, TrainSet, Elm_Type, nHiddenNeurons, ActType,Cs,NL);
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
        X=TestSet(1:j,2:endsize);   %Get all the features
        TempTar=TestSet(1:j,1); %Get the target
        %Format target vector
        Tar=zeros(size(TempTar,1),NofClasses)-1;
        for tempN=1:size(Tar,1)
            Tar(tempN,TempTar(tempN,1))=1;
        end
        clear TempTar;
        H=HOutput(X,IW,Bias',ActType);  %Calculate the hidden layer output
        HS=H;
        TempT=HS*betaS;
        %Formate target
        T=FormatTarget(TempT,1,-1);
        [RSet,Number]=FindZeroRows(T-Tar);
        Error=1-Number/j;%Calculate the error, if it is the first one, e can either be 1 or 0.
        %Store the error
        ESet(j,1)=Error;
        
        clear TempT RSet Number;
        
        %Secondly, calculate the probabilities for labeling
%         TempP=PSSA(x,y,Error);
%         TempP=PSSA(0,0,Error);  %Update based on fixed probability
        %Judge if labeling process is required
        if (ismember(j,LSSet)==0)%No KS selection process
            %Do nothing
            USet=[USet,j];
        else%Labeling process required
            LSet=[LSet,j];
            X=TestSet(j,2:endsize);
            TempTar=TestSet(j,1);
            T=zeros(size(TempTar,1),NofClasses)-1;
            for tempN=1:size(T,1)
                T(tempN,TempTar(tempN,1))=1;
            end
            clear TempTar;
            h=HOutput(X,IW,Bias',ActType);  %Calculate the hidden layer output
            HS=[HS;h];
            Ts=[Ts;T];
            if (length(Ts)>nHiddenNeurons)
                NL=1;
                Case=1;%Flag for incremental learning
            else
                NL=0;
                Case=2;%Flag for inc learning
            end%end if NL
            Elm_Type=1;
            %If this is the first batch, initialize the base classifier
            [IW,Bias,betaS] = ELM_S(TrainSet, TrainSet, Elm_Type, nHiddenNeurons, ActType,Cs,NL);
            
        end %rand(1,1)<=TempP
        clear TempP;
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['ELM-Same-Error-B',num2str(i)];
    LNumSubFolder=['ELM-Same-LNum-B',num2str(i)];
    TimeSubFolder=['ELM-Same-Time-B',num2str(i)];
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

