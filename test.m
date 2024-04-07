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

%% Load Trainednet
load('trained_model.mat');
net = trainednetInfo{1,1};

%% Test Network on One Image

I = readimage(imdsVal,5);
C = semanticseg(I, net);
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',1);
imshow(B)
pixelLabelColorbar(cmap, classes);


%% Evaluate Trained Network
net = trainednetInfo{1,1};

pxdsResults = semanticseg(imdsVal,net, ...
'MiniBatchSize',1, ...
'WriteLocation',tempdir, ...
'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsVal,'Verbose',false);

metrics.DataSetMetrics

metrics.ClassMetrics

% Save in file
% trainednetInfo = {};
% trainednetInfo{1,1} = net;
% trainednetInfo{1,2} = metrics;
% trainednetInfo{1,3} = options;
% save('trained_model.mat','trainednetInfo')