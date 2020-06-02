%This script is used to implement on-line domain adaptation for gas sensor
%array drift compemsation.
%Type: Mainframe  Test
%Coded by :Zhiyuan Ma
%Date: Oct. 2017
%Status: Untested
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc
addpath('ToolFunctions');   %Add fucntion path
%Loading datas
[Data]=LoadFiles();

%Initialize the parameters for experiments
BatchNumber=size(Data,2);
NofClasses=6;
MinNum=50;  %The number of samples used for initial the ELM in target domain

%Parameters of ELM:
nHiddenNeurons=1000;    %The parameters of the hidden layer nodes
ActType='rbf';  %Activation function
Cs=0.001;
Ct=100; %This is Cs in author's method.
Ctu=Cs;

%Parameters of to-be-stored data
Selected=2;    %Number of selected to-be-labeled samples

%Parameters of the results data and files
MainFolder='Data';
SubFolder='Result_AO-DAELM_N1000_';
tail='.txt';
%Begin processing: load each batch in sequence and apply the methods
%accordingly
for Bi=1:2%BatchNumber
    %Judge the cases
    if Bi==1
        %Prepare the data into trainingdata and testing data
        Xs=Data{Bi}(:,2:size(Data{Bi},2));
        Ts=Data{Bi}(:,1);
        TrainingData_File=[Ts,Xs];
        TestingData_File=[Ts,Xs];
        %Set temp paratemter
        if (length(Ts)>nHiddenNeurons)
            NL=1;
        else
            NL=0;
        end%end if NL
        Elm_Type=1;
        %If this is the first batch, initialize the base classifier
        [IW,Bias,betaS] = ELM_S(TrainingData_File, TestingData_File, Elm_Type, nHiddenNeurons, ActType,Cs,NL);
        clear Elm_Type NL; %Release the memory
        
        %Preparing the vectors and matrices for on-line update
        [Hs]=HOutput(Xs,IW,Bias',ActType);
        %Caclulate the output weight matrix beta
        Format_Ts=zeros(size(Ts,1),NofClasses)-1;
        for tempN=1:size(Format_Ts,1)
            Format_Ts(tempN,Ts(tempN,1))=1;
        end %end of for 

        %Calculate the recall rate
        T_sourceP=Hs*betaS;
        Target_sourceP=FormatTarget(T_sourceP,1,-1);
        [RSet_sourceP,No_sourceP]=FindZeroRows(Target_sourceP-Format_Ts); %Calculate the proper classified samples
        NofSamples=size(Format_Ts,1);
        RecallRate=No_sourceP/NofSamples*100;%Calculate the recall rate for each method
        %Calculate the training error
    else        
        %Ramdomize the samples so that they do not arrive in group
        TempData=Data{Bi};
        TempData=TempData(randperm(size(TempData,1)),:);
        
        %Initialize the two variables for storing the number of selected
        %sample and the corresponding accuracy 
        Temp=size(TempData,1);
        Result_Num=zeros(Temp,1);   %Storing the number of selected samples
        Result_Ac=zeros(Temp,1);    %Storing the accuracies after each update
        clear Temp;
        
        %Initialize some temp varables
        NofSamples=size(TempData,1);
        Selected=0;% The number of selected samples that are to be labeled
        RSet=[];%Currrent labeled sample set.
        UIncSet=-1; %Unlabeled increment set for each arrival of samples
        
        %Sequentially processing the data as if they were in a flow
        for Index=2:NofSamples %Index starts from 2 since 1 cannot be used for labeling for KS requires at least 2
            %Initialize current sample that arrives
            x=TempData(Index);
            %Judge if there should be any sample selection
            e=1;%Initially sample error are set to 0;
            y=Index; %Initialize the length of the RSet
            TempP=PSSA(Selected,y,e);
            if (rand(1,1)<=TempP)
                %This means sampling happens
                LabelingFlag=1; %1 for labeling happened and 0 for non.
                %Select possible samples from TempData but excluded from
                %RSet
                if (Selected==0)
                    %This is the first time for selecting samples
                    Selected=2;
                    [model,test]=kenstone(TempData(1:Index,2:size(TempData,2)),Selected);
                    tempc=size(TempData,2);
                    Xt=TempData(model,2:tempc);
                    Tt=TempData(model,1);
                    Xtu=TempData(test,2:tempc);
                    clear tempc;
                    %Randomize the ELM network
                    [IW,Bias]=RandomizeELM(TempData(1,:),NumberofHiddenNeurons,ActType);%we only need the second dimension to help construct ELM
                    %Initialize a target ELM network
                    [Ht]=HOutput(Xt,IW,Bias,ActType);
                    TrainingData_File1=[Tt,Xt];
                    [BetaT]=DA_GIUpdate(TrainingData_File1,Xtu,IW,Bias,ActType,Ct,Ctu,betaS);   %Calculate the output weight matrix
                    continue;
                else
                    %This is not the first time for selecting samples
                    Selected=Selected+1;                
                end%end K==0
                %Perform KS and select the samples that are not included in
                %RSet
                [model,test]=kenstone(TempData(1:Index,2:size(TempData,2)),Selected);
                IncSet=setdiff(model,RSet); %Get the samples that are not included in the IncSet
                
                %Judge if increment is included in the IncSet and update
                %accordingly
                if(ismember(Index,IncSet)==1)   %Only unlabeled incremental happens
                    %No unlabeled incremental happens
                    TempX=TempData(IncSet);
                    h=HOutput(TempX,IW,Bias',ActType);  %Calculate inc labels
                    [beta,K]=LIncUpdate(beta,K,C_T,C_Tu,h,t,HT,HTu,Case); %Only labeled incremental learning
                    clear TempX;
                else
                    %Unlabeled incremental learning happens
                    UIncSet=Index;
                    hx=HOutput(x,IW,Bias',ActType); %Calculate the unlabeled incremental learning
                    [beta,K]=UnIncUpdate(beta,K,hx,HT,betaS,C_Tu,Case);%Unlabeled incremental
                    TempX=TempData(IncSet);
                    hout=HOutput(TempX,IW,Bias',ActType);
                    [beta,K]=UnDecUpdate(beta,K,HT,hout,C_Tu,betaS,Case);%Unlabeled decremental
                    [beta,K]=LIncUpdate(beta,K,C_T,C_Tu,hout,t,HT,HTu,Case);%Labeled incremental
                    clear TempX;
                end %end ismemeber 
            else
                if(Selected==0)
                    %Do nothing since the target classifier has not yet
                    %been initialized
                    continue;
                end
                %This means no labeling
                LabelingFlag=0; %1 for labeling happened and 0 for non.
                
                %Increment of unlabeled sammple triggers unlabeled
                %incremental learning
                UIncSet=Index;
                hx=HOutput(x,IW,Bias,ActType); %Calculate the unlabeled incremental learning
                [beta,K]=UnIncUpdate(beta,K,hx,HT,betaS,C_Tu,Case);%Unlabeled incremental
            end%end if rand(1,1)

            %Store current selected sample number
            TempSize=length(IncSet);%IncSet is the incremental of the selected samples
            Result_Num(Index)=Result_Num(Index-1)+TempSize;
            clear TempSize;
            
            %Calculate the classification performances so far.
            TempX=TempData(1:Index);
            HX=HOutput(TempX,IW,Bias',ActType);
            clear TempX;
            TempT=HX*beta;
            %Format target
            Format_T=zeros(size(TempT,1),NofClasses)-1;
            for tempN=1:size(Format_T,1)
                Format_T(tempN,TempT(tempN,1))=1;
            end %end of for 
            Target_P1=FormatTarget(TempT,1,-1);
            %Calculate the accuracy
            [RSet_sourceP,No_P1]=FindZeroRows(Target_P1-Format_T); %Calculate the proper classified samples
            NofSamples=size(Format_T,1);

            %Store the classification result
            Accuracy=No_P1/NofSamples*100;%Calculate the recall rate for each method
            Result_Ac(Index)=Accuracy;
            %Store the result into variables 
        end %end for Index
        %
        
        %Store the results into file(s) after each batch is finished processing
        %Prepare the name of the files
        NumFileName=[MainFolder,filesep,SubFolder,'b',int2str(Bi),'_Num',tail];   %Result files 
        AcFileName=[MainFolder,filesep,SubFolder,'b',int2str(Bi),'_Ac',tail];
        SaveFiles(Result_Num,NumFileName);
        SaveFiles(Result_Ac,AcFileName);
        
        %Plot the figures of the numbers and the accuracies
        figure;
        plot(Result_Num);
        figure;
        plot(Result_Ac);
        
        %Use the current batch to generate a new base classifier
        %Prepare the data into trainingdata and testing data
        Xs=Data{Bi}(:,2:size(Data{Bi},2));
        Ts=Data{Bi}(:,1);
        TrainingData_File=[Ts,Xs];
        TestingData_File=[Ts,Xs];
        clear Xs Ts;
        %Set temp paratemter
        if (length(Ts)>nHiddenNeurons)
            NL=1;
        else
            NL=0;
        end%end if NL
        Elm_Type=1;
        %If this is the first batch, initialize the base classifier
        [IW,Bias,betaS] = ELM_S(TrainingData_File, TestingData_File, Elm_Type, nHiddenNeurons, ActType,Cs,NL);
        clear Elm_Type NL; %Release the memory
    end%end if Bi
    

end%end for Bi

%To go