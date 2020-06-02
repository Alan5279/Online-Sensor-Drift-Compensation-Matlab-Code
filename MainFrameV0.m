%This manuscript is used to implement the on-line domain adaptation
%methods. All the data are split into training and testing set (See script DatasourceFormat.m)
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
%Begin Testing on training and testing
for i=1:1%length(TrainingCell)    %Change the range for testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    %Randomize TestSet
    TestSet=TestSet(randperm(length(TestSet)),:);
    
    %Initialize base ELM network.
    Ts=TrainSet(:,1);
    if (length(Ts)>nHiddenNeurons)
        NL=1;
    else
        NL=0;
    end%end if NL
    Elm_Type=1;
    %If this is the first batch, initialize the base classifier
    [IW,Bias,betaS] = ELM_S(TrainSet, TrainSet, Elm_Type, nHiddenNeurons, ActType,Cs,NL);
    clear Elm_Type NL; %Release the memory
    
    %Begin Initialize and update target ELM.Starting from 0;
    %In this case, the target ELM  starts with 0 unlabeled data and 1
    %labeled sample.
    for j=1:length(TestSet)
        %Using the first to initialize target ELM
        if j==1
            TempData=TestSet(j,:);
            tempc=size(TempData,2);
            Xt=TempData(:,2:tempc);
            Xtu=0;
            if (length(Xt)>nHiddenNeurons)
                NL=1;
            else
                NL=0;
            end%end if NL
            Elm_Type=1;
            [IW,Bias,betaT] = ELM_S(TempData, TempData, Elm_Type, nHiddenNeurons, ActType,Ct,NL);   %Initialize a new ELM as target ELM
            %Initialize K based on two cases
            Ht=HOutput(Xt,IW,Bias,ActType);
            if(size(Ht,1)>size(Ht,2))
                %More rows than columns
                K=eye(size(Ht,2))+Ct*(Ht'*Ht); 
            else
                %More columns than rows
                K=eye(size(Ht,1))+Ct*(Ht*Ht');
            end%end if size(Ht,1)
            clear tempc Xt Xtu TempData;
            
            
            clear Elm_Type NL;
            %Initialize the parameters
            e=1;    %To be tested whether or not use the residual error of the initialized target ELM
            ULSet=[];    %Initialize the unlabeled set as empty set.
            LSet=[j];   %Initialize labeled set with one labeled sample
            Selected=1; %Initialize the selected samples number
            Htu=[];
        else
            %Start on-line detection and update
            %Judge what kind of incremental/decremental type this is
            tempc=size(TestSet,2);
            x=j;
            y=length(LSet);
            TempP=PSSA(x,y,e);
            clear x y;
            if (rand(1,1)<=TempP)%No KS selection process
                %Put the sample's index j into unlabeled set.
                ULSet=[ULSet,j];
                %Do unlabeled incremental update
                dXtu=TestSet(:,2:tempc);
                h=HOutput(dXtu,IW,Bias',ActType);    %Calculate increment of Htu
%                 Xt=TestSet(LSet,2:tempc);
%                 Ht=HOutput(Xt,IW,Bias,ActType); %Calcualte Ht
                if(size(Ht,1)>=size(Ht,2))   %Ht is updated every time
                    %More rows than columns
                    Case=1;
                else
                    %More columns than rows
                    Case=2;
                end%end if size(Ht,1)
                [betaT,K]=UnIncUpdate(betaT,K,h,Ht,betaS,Ctu,Case);
                Htu=[Htu;h];    %Update Htu;
                clear Case;
            else%Allowing KS process
                %Starting KS process to select more samples for labeling(to do)
                Selected= Selected +1;
                TempSet=TestSet(1:j,2:tempc);
                [model,test]=kenstone(TempSet,Selected);  %KS selection
                IncSet=setdiff(model,LSet); %Get the samples that are not included in the IncSet
                
                %Update Htu
                ULSet=setdiff(ULSet,IncSet);
                TempXtu=TestSet(ULSet,2:tempc);
                Htu=HOutput(TempXtu,IW,Bias',ActType);  %Recalculate Htu
                clear TempXtu;
                
                %Set Case flag
                if (size(Ht,1)+IncSet>=size(Ht,2))
                    Case=1;
                else
                    Case=2;
                end%end if Selected>=size(Ht,2)
                
                
                if(ismember(j,IncSet)==1)
                    %No unlabeled incremental happens
                    TempX=TestSet(IncSet,2:tempc);
                    h=HOutput(TempX,IW,Bias',ActType);  %Calculate inc labels
                    tempt=TestSet(IncSet,1);    %Initialize inc target
                    %Format t
                    t=zeros(size(tempt,1),NofClasses)-1;
                    for tempN=1:size(t,1)
                        t(tempN,tempt(tempN,1))=1;
                    end
                    clear tempt tempN;
                    if(length(IncSet)==1)
                        %Only Labeled incremtnal update requires
                        [betaT,K]=LIncUpdate(betaT,K,Ct,Ctu,h,t,Ht,Htu,Case); %Only labeled incremental learning
                    else
                        %Unlabeled decremental updates 
                        TempDec=setdiff(IncSet,[j]);
                        TempDecX=TestSet(TempDec,2:tempc);
                        hout=HOutput(TempDecX,IW,Bias',ActType);
                        [betaT,K]=UnDecUpdate(betaT,K,Ht,hout,Ctu,betaS,Case);%Unlabeled decremental
                        clear TempDec TempDecX;
                        %Labeled incremental update
                        IncX=TestSet(IncSet,2:tempc);
                        h=HOutput(IncX,IW,Bias',ActType);
                        tempt=TestSet(IncSet,1);
                        t=zeros(size(tempt,1),NofClasses)-1;
                        for tempN=1:size(t,1)
                            t(tempN,tempt(tempN,1))=1;
                        end
                        clear tempt tempN;
                        [betaT,K]=LIncUpdate(betaT,K,Ct,Ctu,h,t,Ht,Htu,Case);%Labeled incremental
                    end%end if length(IncSet)==1
                    
                    clear TempX IncX;
                    %Update Ht
                    Ht=[Ht;h];
                else
                    %j is put into unlabeled and k is in labeled
                    %Unlabeled incremental
                    TempX=TestSet(j,2:tempc);
                    hx=HOutput(TempX,IW,Bias',ActType);
                    [betaT,K]=UnIncUpdate(betaT,K,hx,Ht,betaS,Ctu,Case);%Unlabeled incremental
                    clear TempX;
                    %Note there is a gap between before and after labeling
                    %
                    %Unlabeled decremental 
                    TempDecX=TestSet(IncSet,2:tempc);
                    hout=HOutput(TempDecX,IW,Bias',ActType);
                    [betaT,K]=UnDecUpdate(betaT,K,Ht,hout,Ctu,betaS,Case);%Unlabeled decremental
                    clear TempDecX;
                    %Labeled incremental
                    IncX=TestSet(IncSet,2:tempc);
                    h=HOutput(IncX,IW,Bias',ActType);
                    tempt=TestSet(IncSet,1);
                    t=zeros(size(tempt,1),NofClasses)-1;
                    for tempN=1:size(t,1)
                        t(tempN,tempt(tempN,1))=1;
                    end
                    clear tempt tempN;
                    [betaT,K]=LIncUpdate(betaT,K,Ct,Ctu,h,t,Ht,Htu,Case);%Labeled incremental
                    %Calculate residual error
                    
                    %Update Ht
                    Ht=[Ht;h];
                end%end if ismember(Index,IncSet)
                
                %Increase Labeled Set
                LSet=[LSet,IncSet];
            end%end if rand(1,1)
            clear tempc;
        end%end if j==1
    end
     
end