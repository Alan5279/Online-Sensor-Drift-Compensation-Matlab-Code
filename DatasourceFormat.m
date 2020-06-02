%This script is used for transforming the dataset into 9 training and
%testing files.
%Date: Nov. 2017
%Status: tested
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc
addpath('ToolFunctions');   %Add fucntion path
%Loading data and prepare the training and testing dataset
[Data]=LoadFiles();
TrainingCell=cell(1,9);
TestingCell=cell(1,9);
for i=1:length(Data)-1
    TrainingCell{i}=Data{i};
    TestingCell{i}=Data{i+1};
    %
end%end for
%So far, all the testing and traninig samples have been divided. Note that
%the testing data have not been ramdomized.
