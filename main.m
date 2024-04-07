% clear all
%% Prepare dataset
outputFolder = fullfile(pwd,'uavid_v1.5_official_release_image'); 
labelsZip = fullfile(outputFolder,'labels.zip');
imagesZip = fullfile(outputFolder,'images.zip');

rng(0)
%% Load Images
imgDir = fullfile(outputFolder,'train_data','Images');
imdsTrain = imageDatastore(imgDir);

I = readimage(imdsTrain,1);
I = histeq(I);
imshow(I)

%% Load Pixel-Labeled Images
classes = [
    "Building"
    "Road"
    "StaticCar"
    "Tree"
    "LowVegetation"
    "Human"
    "MovingCar"
    "BackgroundClutter"
    ];

labelIDs = UAVidPixelLabelIDs();

labelDir = fullfile(outputFolder,'train_data','Labels');
pxdsTrain = pixelLabelDatastore(labelDir,classes,labelIDs);

C = readimage(pxdsTrain,1);
cmap = UAVidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B)
pixelLabelColorbar(cmap,classes);

% Prepare dataloader for val and test
imgDir = fullfile(outputFolder,'val_data','Images');
imdsVal = imageDatastore(imgDir);
labelDir = fullfile(outputFolder,'val_data','Labels');
pxdsVal = pixelLabelDatastore(labelDir,classes,labelIDs);

imgDir = fullfile(outputFolder,'test_data','Images');
imdsTest = imageDatastore(imgDir);

%% Analyze Dataset Statistics
tbl = countEachLabel(pxdsTrain);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% Prepare Training, Validation, and Test Sets
numTrainingImages = numel(imdsTrain.Files);
numValImages = numel(imdsVal.Files);
numTestingImages = numel(imdsTest.Files);
% Define validation data.
dsVal = combine(imdsVal,pxdsVal);
dsTrain = combine(imdsTrain,pxdsTrain);

%% Data Augmentation
dsTrain = combine(imdsTrain, pxdsTrain);
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));


%% Create the Network
imageSize = [540 960 3];

% Specify the number of classes.
numClasses = numel(classes);
load('model_architecture.mat')

%% Balance Classes Using Class Weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"labels",pxLayer);
%% Select Training Options


% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',5,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',2.5e-4, ...
    'L2Regularization',0.0001, ...
    'ValidationData',dsVal,...
    'ValidationFrequency',50,...
    'MaxEpochs',40, ...  
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',10,...
    'Plots','training-progress',...
    'OutputNetwork', 'best-validation-loss');

%% Start Training
 
[net, info] = trainNetwork(dsTrain,lgraph,options);



