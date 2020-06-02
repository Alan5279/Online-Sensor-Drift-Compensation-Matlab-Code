%This manuscript is used to test the effectiveness of ODAELMT
%Initialization starts from 0 data.
%Learning: Train on previous and test on the next.
%Status: untested.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc
addpath('ToolFunctions');   %Add fucntion path
%Loading data and prepare the training and testing dataset
[Data]=LoadFiles();
TrainingCell=cell(1,9);
TestingCell=cell(1,9);
for i=1:length(Data)-1
    TrainingCell{i}=Data{i};
    TestingCell{i}=Data{i+1};
    %
end%end for
clear Data;
%The above is from DatasourceFormat.m
%Initialize the parameters
nHiddenNeurons=1000;
ActType='rbf';
Cs=0.1;
Ct=100;
Ctu=Cs;
NofClasses=6; %The number of classes in the table
endsize=size(TrainingCell{1},2);    %Set the end size
%Initialize results
Earray=cell(length(TestingCell));
LNumberarray=cell(length(TestingCell));
LNumber=50;
%Begin training and testing
for i=1:1%For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    %Randomize TestSet
    TestSet=TestSet(randperm(length(TestSet)),:);
    
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
    else
        NL=0;
    end%end if NL
    Elm_Type=1;
    %If this is the first batch, initialize the base classifier
    C=0.1;
    [IW,Bias,betaS] = ELM_S(TrainSet, TrainSet, Elm_Type, nHiddenNeurons, ActType,C,NL);
    clear Elm_Type NL; %Release the memory
    
    %Begin Initialize and update target ELM.Starting from 0;
    %In this case, the target ELM  starts with 0 unlabeled data and 1
    %labeled sample.
    %Initialize the parameters used for update
    beta = betaS;%Initially, use the source classifier for recognition
    HS=HOutput(TrainSet(:,2:endsize),IW,Bias',ActType);
    x=0;y=0;    %x:=the number of total samples; y:= the number of labeled samples
    LSet=[];USet=[];    %LSet:= the set of labeled samples
    %Use KS to label the sample that requires labeling
    [KS,test]=kenstone(TestSet,LNumber);
    for j=1:size(TestSet,1)
        %Calculate the residual error for current iteration
        X=TestSet(1:j,2:endsize);   %Get all the features
        TempTar=TestSet(1:j,1); %Get the target
        %Format target vector
        Tar=zeros(size(TempTar,1),NofClasses)-1;
        for tempN=1:size(Tar,1)
            Tar(tempN,TempTar(tempN,1))=1;
        end
        clear TempTar;
        H=HOutput(X,IW,Bias',ActType);  %Calculate the hidden layer output
        TempT=H*beta;
        %Formate target
        T=FormatTarget(TempT,1,-1);
        [RSet,Number]=FindZeroRows(T-Tar);
        Error=1-Number/length(X(:,1));%Calculate the error, if it is the first one, e can either be 1 or 0.
        ESet(j)=Error;
        
        if (ismember(j,KS))
            %Labeled incremence
            LSet=[LSet,j];%Update the labeled set;
            if (isempty(USet)==0)
                %do nothing;
            else
                %Initialize a base classifier for target domain
                [IW,Bias]=RandomizeELM(TestSet(1,2:endsize),nHiddenNeurons,ActType);%we only need the second dimension to help construct ELM
                Bias=Bias'; %For uniform
                Xt=TestSet(KS,2:endsize);
                Xtu=TestSet(test,2:endsize);
                Ht=HOutput(Xt,IW,Bias',ActType);
                Htu=HOutput(Xtu,IW,Bias',ActType);
                Tempt=TestSet(KS,1);
                %Format t
                Tt=zeros(size(Tempt,1),NofClasses)-1;
                for tempN=1:size(Tt,1)
                    Tt(tempN,Tempt(tempN,1))=1;
                end
                if(size(Ht,1)>=size(Ht,2))%More rows than columsn
                    Case=1;
                else
                    Case=2;
                end
                if(Case==1)
                    %More rows than columns
                    Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
                    K=pinv(eye(size(Temp))+Temp);
                    beta=K*(Ct*Ht'*Tt+Ctu*(Htu'*Htu)*betaS);
                    clear Temp;
                else%Case 2
                    %More columns than rows
                    TempP=Ht*Ht';
                    TempPI=pinv(TempP);
                    TempQ=Ht*Htu';
                    Temp=Ct*TempP+Ctu*TempPI*(TempQ*TempQ');
                    K=pinv(eye(size(Temp))+Temp);
                    beta=Ht'*K*(Ct*Tt+Ctu*TempPI*TempQ*Htu*betaS);
                    clear TempP TempPI TempQ Temp;
                end%end Case==1
            end%end isempty
        else
            %Unlabeled incremence
            USet=[USet,j];
            if(isempty(LSet))
                %Do nothing
            else
                %
                Xt=TestSet(KS,2:endsize);
                Xtu=TestSet(test,2:endsize);
                Ht=HOutput(Xt,IW,Bias',ActType);
                Htu=HOutput(Xtu,IW,Bias',ActType);
                Tempt=TestSet(KS,1);
                %Format t
                Tt=zeros(size(Tempt,1),NofClasses)-1;
                for tempN=1:size(Tt,1)
                    Tt(tempN,Tempt(tempN,1))=1;
                end
                if(size(Ht,1)>=size(Ht,2))%More rows than columsn
                    Case=1;
                else
                    Case=2;
                end
                if(Case==1)
                    %More rows than columns
                    Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
                    K=pinv(eye(size(Temp))+Temp);
                    beta=K*(Ct*Ht'*Tt+Ctu*(Htu'*Htu)*betaS);
                    clear Temp;
                else%Case 2
                    %More columns than rows
                    TempP=Ht*Ht';
                    TempPI=pinv(TempP);
                    TempQ=Ht*Htu';
                    Temp=Ct*TempP+Ctu*TempPI*(TempQ*TempQ');
                    K=pinv(eye(size(Temp))+Temp);
                    beta=Ht'*K*(Ct*Tt+Ctu*TempPI*TempQ*Htu*betaS);
                    clear TempP TempPI TempQ Temp;
                end%end Case==1
            end
        end
        LNumSet(j,1)=length(LSet);
    end%end j=1:size(Test,1)
end%end i=1:1

