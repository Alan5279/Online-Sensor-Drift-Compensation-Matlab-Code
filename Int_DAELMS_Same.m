%This manuscript is used for implement DAELMS
%Status: tested
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    %Randomize TestSet
%     TestSet=TestSet(randperm(length(TestSet)),:);
    [LSSet,TempSet]=kenstone(TestSet(:,2:endsize),50);%LSet is the labeled set 
    clear TempSet;
    
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
    %In this case, the target ELM starts with 0 unlabeled data and 1
    %labeled sample.
    %Initialize the parameters used for update
    HS=HOutput(TrainSet(:,2:endsize),IW,Bias',ActType);
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
        TempT=H*betaS;
        %Formate target
        T=FormatTarget(TempT,1,-1);
        [RSet,Number]=FindZeroRows(T-Tar);
        Error=1-Number/j;%Calculate the error, if it is the first one, e can either be 1 or 0.
        %Store the error
        ESet(j,1)=Error;
        
        clear TempT RSet Number;
        %Judge if labeling process is required
        if (ismember(j,LSSet)==0)%No KS selection process
            %Do nothing
            USet=[USet,j];
        else%Labeling process required
            %Find labeling samples
            LSet=[LSet,j];
            X=TestSet(LSet,2:endsize);
            Tempt=TestSet(LSet,1);
            t=zeros(size(Tempt,1),NofClasses)-1;
            for tempN=1:size(t,1)
                t(tempN,Tempt(tempN,1))=1;
            end
            HT=HOutput(X,IW,Bias',ActType);
            if(Case==1)
                %More rows than columns
                PI=[];
                Temp=Cs*(HS'*HS)+Ct*(HT'*HT);
                K=pinv(eye(size(Temp))+Temp);
                betaS=K*(Cs*HS'*Ts+Ct*HT'*t);
                clear Temp;
            else
                P=HS*HS';
                PI=pinv(P);
                Q=HS*HT';
                Temp=Cs*P+Ct*PI*(Q*Q');
                K=pinv(eye(size(Temp))+Temp);
                betaS=HS'*K*(Cs*Ts+Ct*PI*Q*t);
                clear Temp P;
            end
        end %rand(1,1)<=TempP
        clear TempP;
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['DAELMS-Same-Error-B',num2str(i)];
    LNumSubFolder=['DAELMS-Same-LNum-B',num2str(i)];
    TimeSubFolder=['DAELMS-Same-Time-B',num2str(i)];
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