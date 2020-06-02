%Int_RandomForest_Same
%Status: finished. the algorithm works. Very slow
for i=1:9
    TrainSet=TrainingCell{i};
    TestSet=TestingCell{i};
    ESet=zeros(size(TestSet,1),1);    %Array for storing error
    LNumSet=zeros(size(TestSet,1),1); %Array for storing the number of labeled samples
    TimeSet=zeros(size(TestSet,1),1); %Array for storing the processing time
    %Randomize TestSet
%     TestSet=TestSet(randperm(length(TestSet)),:);
    [LSSet,TempSet]=kenstone(TestSet(:,2:endsize),50);%LSet is the labeled set 
    clear TempSet;
    
    %Set the parameters for randomforest
    nTree=500;
    train_data=TrainSet(:,2:endsize);
    train_label=TrainSet(:,1);
%     test_data=TestSet(:,2:endsize);
    Factor = TreeBagger(nTree, train_data, train_label);    %Train RandomForest
    
    
%     [Predict_label,Scores] = predict(Factor, test_data);
%     Temp=cell2mat(Predict_label);
%     Predict_label=str2num(Temp);
%     [RSet,Number]=FindZeroRows(Predict_label-TestSet(:,1));
%     Accuracy=Number/size(TestSet(:,1),1);
    %Sequentially input the data
    LSet=[];USet=[];    %LSet:= the set of labeled samples
    x=0;y=0;
    for j=1:length(TestSet)
        fprintf('This is the %d sample.\r\n',j);
        tic;%Begining timing
        %Firstly, calculate the current residual classification error rate
        x=j;    %Set x to the number of samples received so far
        %Calculate the residual error
        X=TestSet(1:j,2:endsize);   %Get all the features
        TempTar=TestSet(1:j,1); %Get the target
        %Format target vector
        [Predict_label,Scores] = predict(Factor, X);
        Temp=cell2mat(Predict_label);
        Predict_label=str2num(Temp);
        [RSet,Number]=FindZeroRows(Predict_label-TempTar);
        Error=1-Number/j;%Calculate the error, if it is the first one, e can either be 1 or 0.
        %Store the error
        ESet(j,1)=Error;
        clear TempT RSet Number;
        
%         if (ismember(j,LSSet)==0)%No KS selection process
%             %Do nothing;
%             USet=[USet,j];
%         else
%             LSet=[LSet,j];
%             
%             train_data=[train_data;TestSet(LSet,2:endsize)];
%             train_label=[train_label;TestSet(LSet,1)];
%             Factor = TreeBagger(nTree, train_data, train_label);    %Train RandomForest
%         end
        LNumSet(j,1)=y;
        tempt=toc;%End timing
        TimeSet(j,1)=tempt;
        clear tempt;
    end
    
    %Save the data into files
    ErrorSubFolder=['RandomForest-Error-B',num2str(i)];
    LNumSubFolder=['RandomForest-LNum-B',num2str(i)];
    TimeSubFolder=['RandomForest-Time-B',num2str(i)];
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
