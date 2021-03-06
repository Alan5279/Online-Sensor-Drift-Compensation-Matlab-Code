%This manuscript is for IntegratedExp and the algorithm is LDA
%Status: unfinished.
%Begin training and testing
%Firstly, let us test batch learning;
for i=1:1
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
%     Labels=unique(TrainSet(:,1));
%     LSet=find(TrainSet(:,1)==1);
%     USet=find(TrainSet(:,1)~=1);
%     X=TrainSet(:,2:endsize);
%     Y=[ones(size(TrainSet(LSet,1),1),1);zeros(size(TrainSet(USet,1),1),1)];
    [pc,score,latent,tsquare] = pca(TrainSet(:,2:endsize));
    tran=pc(:,1:12);
    Feature_after_PCA=TrainSet(:,2:endsize)*tran;
    W = LDA(Feature_after_PCA,TrainSet(:,1));
%     W=LDA(X,Y);
%     [pc,score,latent,tsquare] = pca(TestSet(:,2:endsize));
%     tran=pc(:,1:50);
%     Feature_after_PCA=TestSet(:,2:endsize)*tran;
%     TestResult=[ones(size(TestSet,1),1)  Feature_after_PCA]*W';
    TestResult=[ones(size(TrainSet,1),1)  Feature_after_PCA]*W';
    P = exp(TestResult) ./ repmat(sum(exp(TestResult),2),[1 6]);
    P=FormatTarget(P,1,-1);
    Tar=zeros(size(TrainSet,1),6)-1;
    for tempN=1:size(Tar,1)
        Tar(tempN,TrainSet(tempN,1))=1;
    end
    [RSet,Number]=FindZeroRows(P-Tar);
    Accuracy=Number/size(TestSet,1);
%     P = TestResult ./ repmat(sum(TestResult,2),[1 6]);
end

return;

for i=1:9%For testing, only using the first set of training and testing
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
    %Initialize the parameters used for update
    beta=betaS; %
    HS=HOutput(TrainSet(:,2:endsize),IW,Bias',ActType);
    x=0;y=0;    %x:=the number of total samples; y:= the number of labeled samples
    LSet=[];USet=[];    %LSet:= the set of labeled samples
    for j=1:length(TestSet)
        fprintf('This is the %d sample.\r\n',j);
        %The following if is for testing on the performance of batch
        %updating when j=400
        if(j==-200)%Negative means no processing within if else end
            %
            x=j;
            Test=TestSet(1:j,:);
            LNumber=y;
            [KS,test]=kenstone(Test,LNumber);
            LSet=[KS];
            y=length(LSet);
            LNumSet(j)=y;
            USet=[test];
            Xt=Test(KS,2:endsize);
            Xtu=Test(test,2:endsize);
            Ht=HOutput(Xt,IW,Bias',ActType);
            Htu=HOutput(Xtu,IW,Bias',ActType);
            Tempt=Test(KS,1);
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
            continue;
        else
            %Do nothing
        end %j==400
        tic;%Begining timing
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
        TempT=H*beta;
        %Formate target
        T=FormatTarget(TempT,1,-1);
        [RSet,Number]=FindZeroRows(T-Tar);
        Error=1-Number/j;%Calculate the error, if it is the first one, e can either be 1 or 0.
        %Store the error
        ESet(j,1)=Error;
        clear TempT RSet Number;
        
        
        %Secondly, calculate the probabilities of labeling
%         TempP=PSSA(x,y,Error);
%         if(mod(x,200)==0)
%             TempP=1;
%         else
%             TempP=0;
%         end
%         TempP=PSSA(0,0,Error);  %Update based on fixed probability
        %Judge if labeling process is required
        if (ismember(j,LSSet)==0)%No KS selection process
            %Unlabeled incremental learning
            %Judge if there is any labeled samples
            if(y==0)%there is no labeled samples
                %Do nothing. The model, generated in source domain, stays still.
                USet=[USet,j];  %Increase the unlabeled sample;
            else
                %Judge the case value
                if (size(Ht,1)>=size(Ht,2))
                    Case=1; %HT has more rows
                else
                    Case=2;
                end
                %Note that, in this case, there will only be updates for
                %initialization of the network happens only when labeled
                %samples are firstly labeled.
                dx=TestSet(j,2:endsize);
                h=HOutput(dx,IW,Bias',ActType);
                [beta,K]=UnIncUpdate(beta,K,h,Ht,betaS,Ctu,Case);
            end%end y==0
        else%Labeling process required
            %Calculate the IncSet
            %Find labeling samples
            if (x<=2)%There is only one or two samples so far
                %Do nothing, the classfication model stays as the source
                %domain classifier.
                LSet=[LSet,j];
            else
                %There is more than 3 samples samples.
                %Choose the one 
                if (y==0)
                    %Choose the first two of KS algorithm
%                     TempSet=TestSet(1:j,2:endsize);
%                     Selected= 2;
%                     [model,test]=kenstone(TempSet,Selected);
%                     IncSet=model;
%                     LSet=[model];   %Initialize LSet and USet
                    LSet=[j];
                    IncSet=[j];
                    test=USet;
%                     USet=[test];
                    y=1;    %Set the labele sample to 2;
                    clear TempSet;
                    
                    %This is the first time for labeling, therefore the
                    %first time for initialize betaT as well
                    %Firstly, initialize IW and bias for target ELM
                    [IW,Bias]=RandomizeELM(TestSet(1,2:endsize),nHiddenNeurons,ActType);%we only need the second dimension to help construct ELM
                    Bias=Bias'; %For uniform
                    dx=TestSet(IncSet,2:endsize);
                    Tempt=TestSet(IncSet,1);
                    %Format t
                    t=zeros(size(Tempt,1),NofClasses)-1;
                    for tempN=1:size(t,1)
                        Tar(tempN,Tempt(tempN,1))=1;
                    end
                    Ht=HOutput(dx,IW,Bias',ActType);
                    Xtu=TestSet(test,2:endsize);
                    Htu=HOutput(Xtu,IW,Bias',ActType);
                    %Initialize Case
                    if(size(Ht,1)>=size(Ht,2))%More rows than columsn
                        Case=1;
                    else
                        Case=2;
                    end
                    if(Case==1)
                        %More rows than columns
                        Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
                        K=pinv(eye(size(Temp))+Temp);
                        beta=K*(Ct*Ht'*t+Ctu*(Htu'*Htu)*betaS);
                        clear Temp;
                    else%Case 2
                        %More columns than rows
                        TempP=Ht*Ht';
                        TempPI=pinv(TempP);
                        TempQ=Ht*Htu';
                        Temp=Ct*TempP+Ctu*TempPI*(TempQ*TempQ');
                        K=pinv(eye(size(Temp))+Temp);
                        beta=Ht'*K*(Ct*t+Ctu*TempPI*TempQ*Htu*betaS);
                        clear TempP TempPI TempQ Temp;
                    end%end Case==1
                else
                    %Initialize Case
                    if(size(Ht,1)>=size(Ht,2))%More rows than columsn
                        Case=1;
                    else
                        Case=2;
                    end
                    %The betaT has already been initialized, therefore only
                    %update is required
%                     TempSet=TestSet(1:j,2:endsize);
%                     Selected= y+1;
%                     if (Selected>50)
%                         Selected=50;
%                     end
%                     [model,test]=kenstone(TempSet,Selected);
%                     %Calculate IncSet
%                     IncSet=setdiff(model,LSet); %Get the samples that are not included in the IncSet
                    IncSet=[j];
%                     IncSet=IncSet(1);%So far, let us first increase only the first one; Now, testing on 2
%                     if(length(IncSet)<2)
%                         IncSet=IncSet(1);
%                     else
%                         IncSet=IncSet(2);
%                     end
                    LSet=[LSet,IncSet];  
                    y=y+length(IncSet);
                    clear TempSet;   
                    
                    USet=setdiff(USet,IncSet);
                    dx1=TestSet(j,2:endsize);
                    h=HOutput(dx1,IW,Bias',ActType);
                    %Unlabeled incremental first
                    [beta,K]=UnIncUpdate(beta,K,h,Ht,betaS,Ctu,Case);
                    %Unlabeled decremental 
                    dx=TestSet(IncSet,2:endsize);
                    h=HOutput(dx,IW,Bias',ActType);
                    [beta,K]=UnDecUpdate(beta,K,Ht,h,Ctu,betaS,Case);
                    %Update Htu
                    Tempx=TestSet(USet,2:endsize);
                    Htu=HOutput(Tempx,IW,Bias',ActType);
                    %labeling updates
                    Tempt=TestSet(IncSet,1);
                    t=zeros(size(Tempt,1),NofClasses)-1;
                    for tempN=1:size(t,1)
                        t(tempN,Tempt(tempN,1))=1;
                    end
                    [beta,K]=LIncUpdate(beta,K,Ct,Ctu,h,t,Ht,Htu,Case);
                    %Update Ht
                    Tempx=TestSet(LSet,2:endsize);
                    Ht=HOutput(Tempx,IW,Bias',ActType);
                    
                end%end if y==0
            end%end if x<=2
        end%end randperm(1,1)
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end%end j=1:length(TestSet)
    
    %Save the data into files
    ErrorSubFolder=['ODAELMT-Same-Error-B',num2str(i)];
    LNumSubFolder=['ODAELMT-Same-LNum-B',num2str(i)];
    TimeSubFolder=['ODAELMT-Same-Time-B',num2str(i)];
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