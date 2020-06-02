%This manuscript is used for SVM_Same
%Status:tested
K_Type='rbf';
C=0.5;  %Penalty factor?
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
%Begin training and testing
for i=1:9   %For testing, only using the first set of training and testing
    %Prepare training and testing sets
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    [LSSet,TempSet]=kenstone(TestSet(:,2:endsize),50);%LSet is the labeled set 
    
    Xs=TrainSet(:,2:endsize);
    Ts=TrainSet(:,1);
    model=svmtrain(Ts,Xs,'-s 0 -t 2 -c 2 -g 0.02');   %2 is the rbf kernel
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
        T=TestSet(1:j,1);
        [PT,accuracy,decision_values]= svmpredict(T,X,model); %Testing
        Error=1-accuracy(1,1)/100;
        ESet(j,1)=Error;
        
        clear TempT RSet Number;
        %Judge if labeling process is required
        if (ismember(j,LSSet)==0)%No KS selection process
            %Do nothing
            USet=[USet,j];
        else%Labeling process required
            LSet=[LSet,j];
            
            X=TestSet(j,2:endsize);
            T=TestSet(j,1);
            Xs=[Xs;X];
            Ts=[Ts;T];
            model=svmtrain(Ts,Xs,'-s 0 -t 2 -c 2 -g 0.02');   %2 is the rbf kernel
        end %ismember
        
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    %Save the data into files
    ErrorSubFolder=['SVM-Same-Error-B',num2str(i)];
    LNumSubFolder=['SVM-Same-LNum-B',num2str(i)];
    TimeSubFolder=['SVM-Same-Time-B',num2str(i)];
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