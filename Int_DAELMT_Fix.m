%This manuscript is used for DAELMT with fixed period of updating
%Status:unfinished
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
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
%     Flag=0;Selected=50;
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
        if(Flag==Fix)%Labeling required
            Flag=0;%Reset Flag
            if (x<=2)%There is only one or two samples so far
                %Do nothing, the classfication model stays as the source
                %domain classifier.
                LSet=[LSet,j];
            else
                TempSet=TestSet(1:j,2:endsize);
                [model,test]=kenstone(TempSet,Selected);
                IncSet=model;
                LSet=[LSet,model];   %Initialize LSet and USet
                USet=[test];
                if (y==0)
                    y=length(LSet);
                    %This is the first time for labeling, therefore the
                    %first time for initialize betaT as well
                    %Firstly, initialize IW and bias for target ELM
                    [IW,Bias]=RandomizeELM(TestSet(1,2:endsize),nHiddenNeurons,ActType);%we only need the second dimension to help construct ELM
                    Bias=Bias'; %For uniform
                    dx=TestSet(LSet,2:endsize);
                    Tempt=TestSet(LSet,1);
                    %Format t
                    t=zeros(size(Tempt,1),NofClasses)-1;
                    for tempN=1:size(t,1)
                        Tar(tempN,Tempt(tempN,1))=1;
                    end
                    Ht=HOutput(dx,IW,Bias',ActType);
                    Xtu=TestSet(USet,2:endsize);
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
                    y=length(LSet);
                    
                    Xt=TestSet(LSet,2:endsize);
                    Xtu=TestSet(USet,2:endsize);
                    Ht=HOutput(Xt,IW,Bias',ActType);
                    Htu=HOutput(Xtu,IW,Bias',ActType);
                    Tempt=TestSet(LSet,1);
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
                    
                end%end if y==0
            end%end if x<=2
        else
            %Do nothing
            Flag=Flag+1;%Increase Flag
            if(y==0)%there is no labeled samples
                %Do nothing. The model, generated in source domain, stays still.
                USet=[USet,j];
                x=x+1;
            else
                %No labeling process
                USet=[USet,j];
                x=x+1;
                
                Xt=TestSet(LSet,2:endsize);
                Xtu=TestSet(USet,2:endsize);
                Ht=HOutput(Xt,IW,Bias',ActType);
                Htu=HOutput(Xtu,IW,Bias',ActType);
                Tempt=TestSet(LSet,1);
                %Format t
                Tt=zeros(size(Tempt,1),NofClasses)-1;
                for tempN=1:size(Tt,1)
                    Tt(tempN,Tempt(tempN,1))=1;
                end
                %Judge the case value
                if (size(Ht,1)>=size(Ht,2))
                    Case=1; %HT has more rows
                else
                    Case=2;
                end
                %Note that, in this case, there will only be updates for
                %initialization of the network happens only when labeled
                %samples are firstly labeled.
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
            end%end y==0
        end
        
        y=length(LSet);
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['DAELMT-Fix-Error-B',num2str(i)];
    LNumSubFolder=['DAELMT-Fix-LNum-B',num2str(i)];
    TimeSubFolder=['DAELMT-Fix-Time-B',num2str(i)];
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