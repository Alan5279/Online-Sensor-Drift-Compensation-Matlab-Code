%This manuscript is used to implement traditional Component Correction
%based method
%This manuscript is for IntegratedExp and the algorithm is SVM
%Status:tested
TrainingCell=load('Data/RandomD/TraininngCell.mat');
TestingCell=load('Data/RandomD/TestingCell.mat');
TrainingCell=struct2cell(TrainingCell);
TrainingCell=TrainingCell{1};
TestingCell=struct2cell(TestingCell);
TestingCell=TestingCell{1};
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