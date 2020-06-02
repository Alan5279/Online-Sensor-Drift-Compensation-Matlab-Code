%Int_DAELMT_Same
%Status: tested
for i=1:1%For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
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
%     [KS,test]=kenstone(TestSet,LNumber);
    for j=1:size(TestSet,1)
        fprintf('This is the %d th sample.\r\n',j);
        tic;
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
        
        if (ismember(j,LSSet)==0)       
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
                %%%%%The following uses the original function file
                    X_te=TestSet(1:j,2:endsize);
                TestT=TestSet(1:j,1);
                Xt=TestSet(LSet,2:endsize);
                TempTt=TestSet(LSet,1);
                Tt=zeros(size(TempTt,1),NofClasses)-1;
                for tempN=1:size(Tt,1)
                    Tt(tempN,TempTt(tempN,1))=1;
                end
                Ttu=TestSet(USet,1);
                %Set parameters
                [beta]=DAELMT(betaS,IW,Bias',Xt,Tt,Xtu,Ct,Ctu,ActType);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%Use just the formula for update
%                 if(Case==1)
%                     %More rows than columns
%                     Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
%                     K=pinv(eye(size(Temp))+Temp);
%                     beta=K*(Ct*Ht'*t+Ctu*(Htu'*Htu)*betaS);
%                     clear Temp;
%                 else%Case 2
%                     %More columns than rows
%                     TempO=Htu*Ht';
%                     TempTu=Htu*Htu';
%                     TempP=eye(size(TempTu))/Ctu+TempTu;
%                     TempOmega=Ht*Htu';
%                     TempHt=Ht*Ht';
%                     TempR=eye(size(TempHt))/Ct+TempHt;
%                     TempTtu=Htu*betaS;
%                     TempAt=pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*TempTtu-Tt);
%                     TempBtu=pinv(TempP)*TempTtu-pinv(TempP)*TempO*pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*TempTtu-Tt);
%                     beta=Ht'*TempAt+Htu'*TempBtu;
% %                     clear TempP TempPI TempQ Temp;
%                 end%end Case==1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 if(Case==1)
%                     %More rows than columns
%                     Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
%                     K=pinv(eye(size(Temp))+Temp);
%                     beta=K*(Ct*Ht'*Tt+Ctu*(Htu'*Htu)*betaS);
%                     clear Temp;
%                 else%Case 2
%                     %More columns than rows
%                     TempP=Ht*Ht';
%                     TempPI=pinv(TempP);
%                     TempQ=Ht*Htu';
%                     Temp=Ct*TempP+Ctu*TempPI*(TempQ*TempQ');
%                     K=pinv(eye(size(Temp))+Temp);
%                     beta=Ht'*K*(Ct*Tt+Ctu*TempPI*TempQ*Htu*betaS);
%                     clear TempP TempPI TempQ Temp;
%                 end%end Case==1
            end%end y==0
        else%Labeling process required
            %Calculate the IncSet
            %Find labeling samples
            if (x<=2)%There is only one or two samples so far
                %Do nothing, the classfication model stays as the source
                %domain classifier.
                LSet=[LSet,j];
            else
                LSet=[LSet,j];
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
                    Tt=zeros(size(Tempt,1),NofClasses)-1;
                    for tempN=1:size(Tt,1)
                        Tt(tempN,Tempt(tempN,1))=1;
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
                    %%%%%The following uses the original function file
                    X_te=TestSet(1:j,2:endsize);
                TestT=TestSet(1:j,1);
                Xt=TestSet(LSet,2:endsize);
                Tt=TestSet(LSet,1);
                Ttu=TestSet(USet,1);
                TrainingData_File1=[Tt,Xt]; % labeled data in target domain
                TrainingData_File1_tardomain=[Ttu,Xtu]; % unlabeled data in target domain
                TestingData_File1=[TestT,X_te];
                T=TrainingData_File1(:,1);
                Elm_Type=1;
                %Set parameters
                if length(Ts)>nHiddenNeurons; NL=1;else NL=0;end
                if length(Tt)>nHiddenNeurons; NT=1;else NT=0;end
                [IW,Bias,beta]=DAELM_TS(TrainingData_File1,TrainingData_File1_tardomain,TestingData_File1,Elm_Type,nHiddenNeurons,ActType,Cs,Ct,NL,NT,betaS);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%Use just the formula for update
%                 if(Case==1)
%                     %More rows than columns
%                     Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
%                     K=pinv(eye(size(Temp))+Temp);
%                     beta=K*(Ct*Ht'*t+Ctu*(Htu'*Htu)*betaS);
%                     clear Temp;
%                 else%Case 2
%                     %More columns than rows
%                     TempO=Htu*Ht';
%                     TempTu=Htu*Htu';
%                     TempP=eye(size(TempTu))/Ctu+TempTu;
%                     TempOmega=Ht*Htu';
%                     TempHt=Ht*Ht';
%                     TempR=eye(size(TempHt))/Ct+TempHt;
%                     TempTtu=Htu*betaS;
%                     TempAt=pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*TempTtu-Tt);
%                     TempBtu=pinv(TempP)*TempTtu-pinv(TempP)*TempO*pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*TempTtu-Tt);
%                     beta=Ht'*TempAt+Htu'*TempBtu;
% %                     clear TempP TempPI TempQ Temp;
%                 end%end Case==1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
%                     if(Case==1)
%                         %More rows than columns
%                         Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
%                         K=pinv(eye(size(Temp))+Temp);
%                         beta=K*(Ct*Ht'*t+Ctu*(Htu'*Htu)*betaS);
%                         clear Temp;
%                     else%Case 2
%                         %More columns than rows
%                         TempP=Ht*Ht';
%                         TempPI=pinv(TempP);
%                         TempQ=Ht*Htu';
%                         Temp=Ct*TempP+Ctu*TempPI*(TempQ*TempQ');
%                         K=pinv(eye(size(Temp))+Temp);
%                         beta=Ht'*K*(Ct*t+Ctu*TempPI*TempQ*Htu*betaS);
%                         clear TempP TempPI TempQ Temp;
%                     end%end Case==1
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
                    %%%%%The following uses the original function file
                    X_te=TestSet(1:j,2:endsize);
                TestT=TestSet(1:j,1);
                Xt=TestSet(LSet,2:endsize);
                TempTt=TestSet(LSet,1);
                Tt=zeros(size(TempTt,1),NofClasses)-1;
                for tempN=1:size(Tt,1)
                    Tt(tempN,TempTt(tempN,1))=1;
                end
                Ttu=TestSet(USet,1);
                [beta]=DAELMT(betaS,IW,Bias',Xt,Tt,Xtu,Ct,Ctu,ActType);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%Use just the formula for update
%                 if(Case==1)
%                     %More rows than columns
%                     Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
%                     K=pinv(eye(size(Temp))+Temp);
%                     beta=K*(Ct*Ht'*t+Ctu*(Htu'*Htu)*betaS);
%                     clear Temp;
%                 else%Case 2
%                     %More columns than rows
%                     TempO=Htu*Ht';
%                     TempTu=Htu*Htu';
%                     TempP=eye(size(TempTu))/Ctu+TempTu;
%                     TempOmega=Ht*Htu';
%                     TempHt=Ht*Ht';
%                     TempR=eye(size(TempHt))/Ct+TempHt;
%                     TempTtu=Htu*betaS;
%                     TempAt=pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*TempTtu-Tt);
%                     TempBtu=pinv(TempP)*TempTtu-pinv(TempP)*TempO*pinv(TempOmega*pinv(TempP)*TempO-TempR)*(TempOmega*pinv(TempP)*TempTtu-Tt);
%                     beta=Ht'*TempAt+Htu'*TempBtu;
% %                     clear TempP TempPI TempQ Temp;
%                 end%end Case==1
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     if(Case==1)
%                         %More rows than columns
%                         Temp=Ct*(Ht'*Ht)+Ctu*(Htu'*Htu);
%                         K=pinv(eye(size(Temp))+Temp);
%                         beta=K*(Ct*Ht'*Tt+Ctu*(Htu'*Htu)*betaS);
%                         clear Temp;
%                     else%Case 2
%                         %More columns than rows
%                         TempP=Ht*Ht';
%                         TempPI=pinv(TempP);
%                         TempQ=Ht*Htu';
%                         Temp=Ct*TempP+Ctu*TempPI*(TempQ*TempQ');
%                         K=pinv(eye(size(Temp))+Temp);
%                         beta=Ht'*K*(Ct*Tt+Ctu*TempPI*TempQ*Htu*betaS);
%                         clear TempP TempPI TempQ Temp;
%                     end%end Case==1
                    
                end%end if y==0
            end%end if x<=2
        end%end randperm(1,1)
            
            
        LNumSet(j,1)=length(LSet);
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end%end j=1:size(Test,1)
    %Save the data into files
    ErrorSubFolder=['DAELMT-Same-Error-B',num2str(i)];
    LNumSubFolder=['DAELMT-Same-LNum-B',num2str(i)];
    TimeSubFolder=['DAELMT-Same-Time-B',num2str(i)];
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
end%end i=1:1