function [combineAcc rgbAcc depthAcc] = runCRNN()
% init params
params = initParams();
disp(params);

%Initialize Matlab Parallel Computing Enviornment by Xaero | Macro2.cn
if matlabpool('size')<=0
    matlabpool open;
else
    disp('Already initialized');
end

%record the start time
global s;
s = 'Start';
recordtime(s);

%% Run RGB
disp('Forward propagating RGB data');
parmas.depth = false;

% load and forward propagate RGB data
[rgbTrain rgbTest] = forwardProp(params);

% train softmax classifier
disp('Training softmax...');
rgbAcc = trainSoftmax(rgbTrain, rgbTest, params);

%record RGB-Neural&Network time cost
recordtime(strcat(s,'END of Reural&Network'));

%% Run Depth
disp('Forward propagating depth data');
params.depth = true;

% load and forward propagate depth data
[depthTrain depthTest] = forwardProp(params);

% train softmax classifier
depthAcc = trainSoftmax(depthTrain, depthTest,params);

%record Depth-Neural&Network time cost
recordtime(strcat(s,'End of Reural&Network'));

%% Combine RGB + Depth
[cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest);
whos
memory
clear rgbTrain rgbTest depthTrain depthTest;

% test without extra features when combined
params.extraFeatures = false;
combineAcc = trainSoftmax(cTrain, cTest, params);
clear cTrain cTest;

%record Combine-Neural&Network time cost
recordtime(strcat(s,'End of Neural&Network'));

% if matlabpool('size')>0
%     matlabpool close;
% end
return;

function [train test] = forwardProp(params)
global s;
% pretrain filters
recordtime(strcat(s,'Start of Pretraining CNN Filters...'));

% pretrain filters
disp('Pretraining CNN Filters...');
[filters params] = pretrain(params);

recordtime(strcat(s,'End of Pretraining CNN Filters...'));

% forward prop CNN
recordtime(strcat(s,'Start of Forward prop through CNN...'));

disp('Forward prop through CNN...');
[train test] = forwardCNN(filters,params);

clear filters;

recordtime(strcat(s,'End of Forward prop through CNN...'));

% forward prop RNNs
recordtime(strcat(s,'Start of Forward prop through RNN...'));

disp('Forward prop through RNN...');
[train test] = forwardRNN(train, test, params);

recordtime(strcat(s,'End of Forward prop through RNN...'));
return;

function [cTrain cTest] = combineData(rgbTrain, rgbTest, depthTrain, depthTest)
% ensure they come from the same file
testCompatability(rgbTrain, depthTrain);
testCompatability(rgbTest, depthTest);

% combine data
cTrain.data = [rgbTrain.data; depthTrain.data];
cTest.data = [rgbTest.data; depthTest.data];

% normalize depth and rgb features independently
m = mean(cTrain.data,2);
s = std(cTrain.data,[],2);
cTrain.data = bsxfun(@rdivide, bsxfun(@minus, cTrain.data,m),s);
cTest.data = bsxfun(@rdivide, bsxfun(@minus, cTest.data,m),s);

% add the labels
cTrain.labels = rgbTrain.labels;
cTest.labels = rgbTest.labels;

clear m s;
return;

function testCompatability(rgb, depth)
assert(length(rgb.file) == length(depth.file));
parfor i = 1:length(rgb.file)
    assert(strcmp(rgb.file{i}, depth.file{i}));
end

assert(isequal(rgb.labels, depth.labels));
assert(isequal(rgb.labels, depth.labels));
return

function disp_str = recordtime(str)
disp_str = datestr(now,'yyyy-mm-dd HH:MM:SS');
disp_str = sprintf('%s Running At: %s',str,disp_str);
disp(disp_str);
return
