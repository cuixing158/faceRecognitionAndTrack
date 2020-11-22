
%% 训练脚本
% arcface+mobilenet， 自定义arcFace层

%%
addpath('./utils/');
inputSize = [112,112];
embedding_size = 512;

imds = imageDatastore('../dataSets/facebank/',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames',...
    'FileExtensions',{'.jpg','.png'});
imds.ReadFcn = @(x)preprocessTrain(x,inputSize);
numClasses = length(categories(imds.Labels));

%% network arc
backbone = mobilenetv2();
lg = layerGraph(backbone);
lg = removeLayers(lg,{'input_1',...
    'global_average_pooling2d_1',...
    'Logits','Logits_softmax',...
    'ClassificationLayer_Logits'});

newLayers = [globalAveragePooling2dLayer('name','avg'),...
    fullyConnectedLayer(embedding_size,'name','embedding','Bias',zeros(embedding_size,1,'single'),'BiasLearnRateFactor',0),... % bias 置为0
    arcfaceLayer("arc_face",embedding_size, numClasses),... % 此种方式是基于layerGraph搭建网络，自定义arcFace
    softmaxLayer('name','softmax'),...
    classificationLayer('name','cls')];

inputLayer = imageInputLayer([inputSize,3],'name','input');
lg = addLayers(lg,inputLayer);
lg = addLayers(lg,newLayers);
lg = connectLayers(lg,'input','Conv1');
lg = connectLayers(lg,'out_relu','avg');
analyzeNetwork(lg)

%% 冻结参数+train
lg = freezeWeights(lg,150);
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',24, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imds,lg,options);
save('../models/faceRecogPth/mobilenetv2_arcface.mat','net')




