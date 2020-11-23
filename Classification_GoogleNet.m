clc;
clear all;
close all;
net = googlenet;
%%
imds = imageDatastore('EDATA_SEG','IncludeSubfolders',true,'LabelSource','foldernames');
%%
%Divide the data into 70% training data and 30% validation data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
inputSize = net.Layers(1).InputSize;
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%%
layers=net.Layers
%%
%Resize images to match the pretrained network input size.
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,{'pool5-drop_7x7_s1','loss3-classifier','prob','output'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    dropoutLayer(0.6,'Name','newDropout')
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-7x7_s1','newDropout');
inputSize = net.Layers(1).InputSize;
%%

%%
% Specify training options.
% Specify the mini-batch size, that is, how many images to use in each iteration.
% Specify a small number of epochs. An epoch is a full training cycle on the entire training data set. For transfer learning, you do not need to train for as many epochs. Shuffle the data every epoch.
% Set InitialLearnRate to a small value to slow down learning in the transferred layers.
% Specify validation data and validation frequency so that the accuracy on the validation data is calculated once every epoch.
% Turn on the training plot to monitor progress while you train.
miniBatchSize = 10;
% valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%%
trainedNet = trainNetwork(augimdsTrain,lgraph,options);
%%
[YPred,scores] = classify(trainedNet,augimdsValidation);
%%
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
%%
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
%%
plotconfusion(YValidation,YPred)