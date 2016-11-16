%implimentation for SDG Stochastic gradient Descent 
%by liyize 2016 11 14
%http://lamda.nju.edu.cn/yehj/DM16/dm16.html
clear;
data1TestPath = 'data/data1/dataset1-a9a-testing.txt';
data1TrainPath = 'data/data1/dataset1-a9a-training.txt';
data2TestPath = 'data/data2/covtype-testing.txt';
data2TrainPath = 'data/data2/covtype-training.txt';

 data1TestMatrix = csvread(data1TestPath);
 data1TrainMatrix = csvread(data1TrainPath);
 data2TestMatrix = csvread(data2TestPath);
 data2TrainMatrix = csvread(data2TrainPath);
 
[data1TrainM,~] =size(data1TrainMatrix);
[data2TrainM,~] =size(data2TrainMatrix);
numToGo1 = int32( data1TrainM*0.2);
numToGo2 = int32( data2TrainM*0.2);


% method1 data1 build B 
stepSize1 = 0.01;
bw1 = 0.01;
tic;
B11 = SDG_Train1(stepSize1,bw1,data1TrainMatrix,data1TestMatrix,numToGo1,1);
toc;
% method1 data2 build B 
stepSize2 = 0.005;
bw1 = 0.0005;
tic;
B12 = SDG_Train1(stepSize2,bw1,data2TrainMatrix,data2TestMatrix,numToGo2,2);
toc;

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% method2 data1 build B 
stepSize3 = 0.005;
bw2 = 0.00005;
tic;
B21 = SDG_Train2(stepSize3,bw2,data1TrainMatrix,data1TestMatrix,numToGo1,1);
toc;
%method2 data2 build B 
stepSize4 = 0.005;
bw2 = 0.000001;
tic;
B22 = SDG_Train2(stepSize4,bw2,data2TrainMatrix,data2TestMatrix, numToGo2,2);
toc;







