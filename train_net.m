%% Train

clear; clc ;
rng default;

% Load the image dataset

imds = imageDatastore('image/CK+/', 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imds = shuffle(imds);

[imdsTrain, imdsValid, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15);

net = mobilenetv2;
layers = net.Layers;
inputSize = net.Layers(1).InputSize;

% Image preprocessing
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

processTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');

processValid = augmentedImageDatastore(inputSize(1:2), imdsValid, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');

processTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing','gray2rgb');


numClasses = numel(categories(imdsTrain.Labels));

% Transfer the network

lgraph = layerGraph(net);
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'Logits',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',newClassLayer);

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment', 'gpu', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs',300, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',processValid, ...
    'ValidationFrequency', 60, ...
    'Verbose', true, ...
    'Plots','training-progress', ...
    'OutputNetwork', 'best-validation-loss');

model = trainNetwork(processTrain, lgraph, options);

save('model/mobile_latest.mat', 'model');

%% Evluation

[YPred, scores] = classify(model, imdsTest);

idx = randperm(numel(imdsTest.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest);

figure
plotconfusion(imdsTest.Labels, YPred);

