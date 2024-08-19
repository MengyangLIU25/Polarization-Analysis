clear all 
clc
%% 

SAVE = false;   
NN = 0.7;  
%% 

sample_name{1} = '';
sample_name{2} = '';
sample_name{3} = '';
sample_name{4} = '';
sample_name{5} = '';
sample_name{6} = '';

[~,len_f]=size(sample_name);

%% 

mmt_par_T{1} = xlsread('');
mmt_par_T{2} = xlsread('');
mmt_par_T{3} = xlsread('');
mmt_par_T{4} = xlsread('');
mmt_par_T{5} = xlsread('');
mmt_par_T{6} = xlsread('');

Ang = 0;

[~,tti]=size(mmt_par_T);
[~,N] = size(mmt_par_T{1});
%% 

 for i = 1:1:tti
    for j = 1:1:N
        TF = isoutlier(mmt_par_T{i}(:,j),'quartiles'); 
        F = find (TF==1);
        mmt_par_T{i}(F,:) = [];
    end
 end


for i = 1:1:tti
    hh(i) = size(mmt_par_T{i},1);
end
num1 = floor(min(hh)); 

N_= N+1; 
num2 = tti*num1; 

%%
for i = 1:1:tti    
%     data((ti-1)*num1+1:ti*num1,1:N)=mmt_par_T{ti}(1:num1,Ang*6+1:Ang*6+N);
    num = size(mmt_par_T{i},1);
    if num > num1
        [a] = randperm(num,num1);
        data((i-1)*num1+1:i*num1,1:N)=mmt_par_T{i}(a,1:N);
    else   
        data((i-1)*num1+1:i*num1,1:N)=mmt_par_T{i}(1:num1,1:N);
    end
    data((i-1)*num1+1:i*num1,N_)=i; 
end

k=rand(1,num2);
[m,n]=sort(k);

input=data(:,1:N);
output1 =data(:,N_);

output=zeros(num2,tti);

for i =  1:1:tti
    st = num1*(i-1)+1;
    output(st:num1*i,i)= output(st:num1*i,i)+1;
end

%% 
num3 = floor(NN* num2);
input_train=input(n(1:num3),:)';
output_train=output(n(1:num3),:)';
input_test=input(n(num3+1:end),:)';
output_test=output(n(num3+1:end),:)';
num4 = length(input_test);
 train_label = output1(n(1:num3),1)';
 test_label =  output1(n((num3+1):end),1)';

[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% 

net=newff(inputn,outputn,[16 20 20 16]);
    
net.trainParam.epochs = 10000;
net.trainParam.lr=0.01;

net.trainParam.goal=0.001;
    
net=train(net,inputn,outputn,'useGPU','yes');
% net=train(net,inputn,outputn);
if SAVE
    save(mynet,"net","outputps","inputps")
end

an=sim(net,inputn);
BPoutput_train=mapminmax('reverse',an,outputps);
outputtrain_fore=zeros(1,num3);

for i=1:num3
    outputtrain_fore(i)=find(BPoutput_train(:,i)==max(BPoutput_train(:,i)));
end
% confusion_matrixall(train_label,outputtrain_fore);
acc1 = 0;
for i=1:num3
    if outputtrain_fore(i) == train_label(i)
    acc1 = acc1 + 1;
    end
end
acc1 = acc1 / num3;
fprintf('Training set Accuracy is %f\n', acc1);

%% BPNN
inputn_test=mapminmax('apply',input_test,inputps);
 
an=sim(net,inputn_test);

BPoutput_test=mapminmax('reverse',an,outputps);

outputtest_fore=zeros(1,num4);
for i=1:num4
    outputtest_fore(i)=find(BPoutput_test(:,i)==max(BPoutput_test(:,i)));
end

acc2 = 0;
for i=1:num4
    if outputtest_fore(i) == test_label(i)
    acc2 = acc2 + 1;
    end
end
acc2 = acc2 / num4;
acc2 = acc2*100;
acc3 = round(acc2,1);

numFeatures = size(input_train, 2);

sensitivity = zeros(1, numFeatures);

for i = 1:numFeatures
    inputWithFeature = input_train;
    inputWithoutFeature = input_train;
    inputWithoutFeature(:, i) = 0; 

    outputWithFeature = sim(net, inputWithFeature');
    outputWithoutFeature = sim(net, inputWithoutFeature');

    deltaOutput = outputWithFeature - outputWithoutFeature;

    sensitivity(i) = mean(abs(deltaOutput));
end

disp(sensitivity);

fprintf('Test set Accuracy is %f\n', acc3);
confusion_matrixall(test_label,outputtest_fore,sample_name,acc3); 
