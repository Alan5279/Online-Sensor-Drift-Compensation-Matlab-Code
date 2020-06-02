%This function returns more result than base classifiers
%% Weighted ELM method
function [InputWeight,BiasofHiddenNeurons,OutputWeight] = ELM_(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,C,NL)


%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
train_data = TrainingData_File;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data =  TestingData_File;
TV.T=test_data(:,1)';
TE0=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array


NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

[x,posx1] = find(T==1);[x,posx2] = find(T==2);[x,posx3] = find(T==3);[x,posx4] = find(T==4);
[x,posx5] = find(T==5);[x,posx6] = find(T==6);
W = zeros(length(T),length(T));
AVG=length(T)/6;
% W--1 method
%{
for i = 1:length(T)
    if T(i) == 1
        if length(posx1)>AVG
           W(i,i)= 0.618/length(posx1);
        else W(i,i)= 1/length(posx1);
        end
    elseif T(i)==2
        if length(posx2)>AVG
            W(i,i)= 0.618/length(posx2);
        else W(i,i)= 1/length(posx2);
        end
    elseif T(i)==3
        if length(posx3)>AVG
            W(i,i)= 0.618/length(posx3);
        else W(i,i)= 1/length(posx3);
        end
    elseif T(i)==4
        if length(posx4)>AVG
            W(i,i)= 0.618/length(posx4);
        else W(i,i)= 1/length(posx4);
        end
    elseif T(i)==5
        if length(posx5)>AVG
            W(i,i)= 0.618/length(posx5);
        else W(i,i)= 1/length(posx5);
        end
    elseif T(i)==6
        if length(posx6)>AVG
            W(i,i)= 0.618/length(posx6);
        else W(i,i)= 1/length(posx6);
        end
    end 
end
%}
% W--2 method
%
for i = 1:length(T)
    if T(i) == 1
        W(i,i)= 1;%length(posx1)/length(T);%1/length(posx1);
    elseif T(i)==2
        W(i,i)= 1;%length(posx2)/length(T);%1/length(posx2);
    elseif T(i)==3
        W(i,i)= 1;%length(posx3)/length(T);%1/length(posx3);
    elseif T(i)==4
        W(i,i)= 1;%length(posx4)/length(T);%1/length(posx4);
    elseif T(i)==5
        W(i,i)= 1;%length(posx5)/length(T);%1/length(posx5);
    elseif T(i)==6
        W(i,i)= 1;%length(posx6)/length(T);%1/length(posx6);
    end
end
%}
%save W W;

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
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
    number_class=6;
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
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas','rbf'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
H=H';T=T';
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% OutputWeight=pinv(H') * T';                        % slower implementation
% n = size(T,2);
% OutputWeight=H'*((H'*H+speye(n)/C)\(T')); 

if NL==1 % 样本数N大于隐层神经元数L
   n = NumberofHiddenNeurons;
   OutputWeight=((H'*W*H+speye(n)/C)\(H'*W*T)); 
else % 样本数N小于隐层神经元数L
   n = size(T,1);
   OutputWeight=H'*((W*H*H'+speye(n)/C)\(W*T)); 
end