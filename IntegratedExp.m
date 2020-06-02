%This manuscript is used for integrated experiments
%For Standard PSSA
clear;clc
addpath('ToolFunctions');   %Add fucntion path
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
K_Type='rbf';
C=0.5;  %Penalty factor?
Int_ODAELMT;
% 
Int_ODAELMS;
% 
Int_DAELM;%This is DAELM-T

Int_ELM;%%

Int_SVM;

% Int_Ensemble_ELM;

% Int_Ensemble_SVM;

Int_DAELMS;

% Int_RandomForest;