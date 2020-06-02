%This manuscript is used to implement traditional ELM
%This manuscript is for IntegratedExp and the algorithm is ELM
%Status:tested
%Begin training and testing
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
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
                    Tempt=TestSet(IncSet,1);
                    %Format t
                    t=zeros(size(Tempt,1),NofClasses)-1;
                    for tempN=1:size(t,1)
                        Tar(tempN,Tempt(tempN,1))=1;
                    end
                    h=HOutput(dx,IW,Bias',ActType);
                    HT=h;
                    if(Case==1)
                        %More rows than columns
                        PI=[];
                        HS=[HS;HT];
                        Ts=[Ts;t];
                        betaS=pinv(HS'*HS)*HS'*Ts;
%                         Temp=Cs*(HS'*HS)+Ct*(h'*h);
%                         K=pinv(eye(size(Temp))+Temp);
%                         betaS=K*(Cs*HS'*Ts+Ct*h'*t);
%                         clear Temp;
                    else
                        HS=[HS;HT];
                        Ts=[Ts;t];
                        betaS=HS'*pinv(HS*HS')*Ts;
%                         P=HS*HS';
%                         PI=pinv(P);
%                         Q=HS*h';
%                         Temp=Cs*P+Ct*PI*(Q*Q');
%                         K=pinv(eye(size(Temp))+Temp);
%                         betaS=HS'*K*(Cs*Ts+Ct*PI*Q*t);
%                         clear Temp P;
                    end
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
                        HS=[HS;HT];
                        Ts=[Ts;t];
                        betaS=pinv(HS'*HS)*HS'*Ts;
%                         Temp=Cs*(HS'*HS)+Ct*(HT'*HT);
%                         K=pinv(eye(size(Temp))+Temp);
%                         betaS=K*(Cs*HS'*Ts+Ct*HT'*t);
%                         clear Temp;
                    else
                        HS=[HS;HT];
                        Ts=[Ts;t];
                        betaS=HS'*pinv(HS*HS')*Ts;
%                         P=HS*HS';
%                         PI=pinv(P);
%                         Q=HS*HT';
%                         Temp=Cs*P+Ct*PI*(Q*Q');
%                         K=pinv(eye(size(Temp))+Temp);
%                         betaS=HS'*K*(Cs*Ts+Ct*PI*Q*t);
%                         clear Temp P;
                    end
                    
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
    ErrorSubFolder=['ELM-Error-B',num2str(i)];
    LNumSubFolder=['ELM-LNum-B',num2str(i)];
    TimeSubFolder=['ELM-Time-B',num2str(i)];
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