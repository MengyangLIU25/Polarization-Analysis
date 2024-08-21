%% Clear environment variables
clc;clear;close all

%% Data Import
data_input = {};

data_input{1} = readmatrix('PP-L 0.2-1 um-2-45pz.xlsx');
data_input{2} = readmatrix('PP-L 1-5 um-2-45pz.xlsx');
data_input{3} = readmatrix('PP-L 5-20 um-45pz.xlsx');
data_input{4} = readmatrix('PP-L 20-40 um-3-45pz.xlsx');
data_input{5} = readmatrix('PP-L 40-60 um-45pz.xlsx');
data_input{6} = readmatrix('PP-L Smaller than 60 um-45pz.xlsx');


%% Hyperparameters
hidden_layer_sizes = [32 16 8];
epochs = 2000;
learning_rate = 0.1;
goal = 1e-5;
train_ratio = 0.7;
val_ratio = 0.15;
test_ratio = 0.15;
max_fail = 6;
setdemorandstream(2);

%% Preprocess input data
num_classes = length(data_input);
labels = linspace(1,num_classes,num_classes);
all_data = [];
all_labels = [];

for i = 1:num_classes
     data = data_input{i};

     num_samples = size(data,1);
     label = repmat(labels(i),num_samples,1);

     all_data = cat(1,all_data,data);
     all_labels = cat(1,all_labels,label);
end

[X,dataps] = mapminmax(all_data);
y = zeros(length(all_labels), num_classes);
for i = 1:length(y)
    y(i, all_labels(i)) = 1;
end

%% Training process

net=feedforwardnet(hidden_layer_sizes);

net.trainParam.epochs = epochs;
net.trainParam.lr=learning_rate;
net.trainParam.goal = goal;
net.trainFcn = 'trainscg';  

net.divideParam.trainRatio = train_ratio; 
net.divideParam.valRatio = val_ratio; 
net.divideParam.testRatio = test_ratio;
net.trainparam.max_fail =max_fail; 

[net, tr,net_y] = train(net, X', y');

testInd = tr.testInd;
test_pred = net(X(testInd,:)');
[~, predicted_classes] = max(test_pred, [], 1);
[~, actual_classes] = max(y(testInd,:), [], 2);
accuracy = sum(predicted_classes' == actual_classes) / length(actual_classes);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

conf_matrix = confusionmat(actual_classes, predicted_classes');
class_labels = {'0.2-1μm','1-5μm','5-20μm','20-40μm','40-60μm','Mix'};
confusionchart(conf_matrix,class_labels);
