%This manuscript is used to implement the randomize of the input layer and
%output weight of ELM
function [IW,Bias]=RandomizeELM(Pattern,nHiddenNeurons,ActivationFunction)
     %Randomize ELM parameters
    InputNeurons= size(Pattern,2);    %Get the number of the input neurons
    IW = rand(nHiddenNeurons, InputNeurons)*2-1;
    switch lower(ActivationFunction)
        case{'rbf'}
            Bias = rand(1,nHiddenNeurons);
%             Bias=ones(1,nHiddenNeurons);
%           Bias = rand(1,nHiddenNeurons)*1/3+1/11;     %%%%%%%%%%%%% for the cases of Image Segment and Satellite Image
%           Bias = rand(1,nHiddenNeurons)*1/20+1/60;    %%%%%%%%%%%%% for the case of DNA
        case{'sig'}
            Bias = rand(1,nHiddenNeurons)*2-1;
        case{'sin'}
            Bias = rand(1,nHiddenNeurons)*2-1;
        case{'hardlim'}
            Bias = rand(1,nHiddenNeurons)*2-1;
    end
end