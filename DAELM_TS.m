%% Target domain transfer ELM (model 1 trained on labeled target domain)
function [InputWeight, BiasofHiddenNeurons,OutputWeight] = DAELM_TS(TrainingData_File,TrainingData_File_tardomain, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,Cs,Ct,NL,NT,BetaS)
% This is an implementation of the algorithm for learning the
% DAELM from a labeled source domain data and a target domain data
%
% Please refer to the following paper
% Lei Zhang and David Zhang,"Domain Adaptation Extreme Learning Machines for Drift Compensation", In IEEE Transactions on Instrumentation & Measurement, 2015.

% Input:
%      (1)TrainingData_File=[Ts,Xs]; % Training data of source domain
%          Ts:Nx1 (training label in source domain);Xs:Nxd(training matrix in source domain)
%      (2)TrainingData_File_tardomain=[Tt,Xt]; % training data of target domain
%          Tt:Ntx1(training label in target domain);Xt:Ntxd(training matrix in target domain)
%      (3)TestingData_File=[TestT,X_te]; % testing data of target domain
%          TestT:Ntex1(testing label in target domain);X_te:Ntexd(testing matrix in target domain)
%      (4)Elm_Type:0 or 1 (regression or classifier)
%      (5)NumberofHiddenNeurons: 1000
%      (6)ActivationFunction:radbas
%      (7)Cs:regularization coefficient
%      (8)Ct:regularization coefficient
%      (9)NL:model selection
%      (10)NT:model selection
%      (11)BetaS:base classifier
% Output:
%      (1)MissClassificationRate_Training: number of wrong classification of training data
%      (2)MissClassificationRate_Testing: number of wrong classification of testing data

% please contact us via the following email:
% email:leizhang@cqu.edu.cn

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset i.e. souce domain
train_data = TrainingData_File;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%% Load training data in target domain
train_target_data=TrainingData_File_tardomain;
Tt=train_target_data(:,1)';
Pt=train_target_data(:,2:size(train_target_data,2))';

%%%%%%%%%%% Load testing dataset
test_data =  TestingData_File;
TV.T=test_data(:,1)';
TE0=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTrainingData_Target=size(Pt,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

%{
[x,posx1] = find(T==1);[x,posx2] = find(T==2);[x,posx3] = find(T==3);[x,posx4] = find(T==4);
[x,posx5] = find(T==5);[x,posx6] = find(T==6);
W = zeros(length(T),length(T));
% weight method
for i = 1:length(T)
    if T(i) == 1
        W(i,i)= 1;%1/length(posx1);
    elseif T(i)==2
        W(i,i)= 1;%1/length(posx2);
    elseif T(i)==3
        W(i,i)= 1;%1/length(posx3);
    elseif T(i)==4
        W(i,i)= 1;%1/length(posx4);
    elseif T(i)==5
        W(i,i)= 1;%1/length(posx5);
    elseif T(i)==6
        W(i,i)= 1;%1/length(posx6);
    end 
end
%}

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
%     sorted_target=sort(cat(2,T,TV.T),2);
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    number_class=6;% the number of class
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
    
    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
% start_time_train=cputime;
tic;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
tempHt=InputWeight*Pt;
clear P;                                            %   Release input of training data 
clear Pt;
ind=ones(1,NumberofTrainingData);
indt=ones(1,NumberofTrainingData_Target);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
BiasMatrixT=BiasofHiddenNeurons(:,indt);
tempH=tempH+BiasMatrix;
tempHt=tempHt+BiasMatrixT;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
        Ht=1./(1+exp(-tempHt));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
        Ht = sin(tempHt);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
        Ht = double(hardlim(tempHt));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
        Ht = tribas(tempHt);
    case {'radbas','rbf'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        Ht = radbas(tempHt);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
clear tempHt;

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% OutputWeight=pinv(H') * T';                        % slower
% implementation
% n = size(T,2);
% OutputWeight=H'*((H'*H+speye(n)/C)\(T')); 
n = NumberofHiddenNeurons;
% OutputWeight=((H*W*H'+speye(n)/C)\(H*W*T')); 
% OutputWeight=mtimesx(H,((mtimesx(H',H)+speye(n)/C)\T')); 
% OutputWeight=inv(H * H') * H * T';                         % faster implementation

%% Domain Adaption ELM
H=H';
Ht=Ht';
T=T';%Tt=Tt';
Tt=Ht*BetaS;
if NT==0 % 样本数N小于隐层神经元数L
   A=Ht*H';
   B=Ht*Ht'+speye(NumberofTrainingData_Target)/Ct;
   C=H*Ht';
   D=H*H'+speye(NumberofTrainingData)/Cs;
   AlphaT=inv(B)*Tt-inv(B)*A*inv(C*inv(B)*A-D)*(C*inv(B)*Tt-T);
   AlphaS=inv(C*inv(B)*A-D)*(C*inv(B)*Tt-T);
   OutputWeight=H'*AlphaS+Ht'*AlphaT;
else % 样本数N大于隐层神经元数L 
    OutputWeight=inv(speye(n)+Cs*H'*H+Ct*Ht'*Ht)*(Cs*H'*T+Ct*Ht'*Tt);
end

%%%%%%%%%%% Calculate the training accuracy
Y=(H * OutputWeight);                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))  ;             %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas','rbf'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test      ;     %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY))       ;     %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
           MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
     
% testing
    MissClassificationRate_Testing=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
           MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
end