%This manuscript is used for implement the experiments using the same
%updating sequence 

clear;clc
addpath('ToolFunctions');   %Add fucntion path
addpath('LDA');
%Loading data and prepare the training and testing dataset
% [Data]=LoadFiles();
% TrainingCell=cell(1,9);
% TestingCell=cell(1,9);
% for i=1:length(Data)-1
%     TrainingCell{i}=Data{i};
%     TempSet=Data{i+1};
%     TempSet=TempSet(randperm(length(TempSet)),:);%Randomize tempset
%     TestingCell{i}=TempSet;
%     clear TempSet;
%     %
% end%end for
% clear Data;
%The above is from DatasourceFormat.m
DataFile=['Data',filesep,'RandomD',filesep];
TrainingName='TrainingCell';
TestingName='TestingCellR.mat';
Test=load([DataFile,TrainingName]);
TrainingCell=Test.TrainingCell;
Test=load([DataFile,TestingName]);
TestingCell=Test.TestingCell;

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
Period=500;

%Record the sampling for update
%DAELMS
% Int_DAELMS_Same;

%DAELMT
% Int_DAELMT_Same2;

% Int_ODAELMS_Same;
Int_ODAELMS_Same2;

%ODAELMT
% Int_ODAELMT_Same2;

%ELM
% Int_ELM_Same;

%SVM
% Int_SVM_Same;

%SVM ensemble
% Int_Ensemble_SVM_Same;

%ELM_Ensemble_Same
% Int_Ensemble_ELM_Same;

% Int_LDA_Same;%Too low accuracy

%RandomForest
% Int_RandomForest_Same;

