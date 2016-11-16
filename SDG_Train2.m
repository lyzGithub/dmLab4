function [ re ] = SDG_Train2( step, bw, dataTrainMatrix, dataTestMatrix,numToGo,dataType)
%µü´úbuild classffier B
%step is step£¬ bw is the balance 
%B first random build
%   Detailed explanation goes here
disp(['here to go method2 data',num2str(dataType)]);
[~,dataTestN] = size(dataTestMatrix);
[dataTrainM,dataTrainN] = size(dataTrainMatrix);
dataTestLabel = dataTestMatrix(:,dataTestN);
dataTrainLabel = dataTrainMatrix(:,dataTrainN);
dataTestMatrix(:, dataTestN) = [];
dataTrainMatrix(:,dataTrainN) = [];

B = rand(dataTestN-1,1)';

resultComMy = zeros(100,1);
errorRateMyTrain = zeros(numToGo,1);
errorRateMyTest = zeros(numToGo,1);
numToGoVector = zeros(numToGo,1);
deb = int32(numToGo/100);
j = 1;
for i = 1:numToGo
    randomi = randi(dataTrainM);
    randomXi = dataTrainMatrix(randomi,:);
    randomLabeli = dataTrainLabel(randomi,1);
    deta = (-randomXi) * 2 * (randomLabeli -  B * (randomXi') ) + bw* 2*B;
    B = B - step * deta ;
    if ( (j-1)*deb + 1  )== i
        resultComMy(j) =  mean((dataTrainLabel-dataTrainMatrix*(B')).^2) + bw*norm(B, 2);
        j = j+1;
    end
    errorRateMyTrain(i) = predict_error_rate(B,dataTrainLabel,dataTrainMatrix);
    errorRateMyTest(i) = predict_error_rate(B,dataTestLabel,dataTestMatrix);
    numToGoVector(i) = i;
end

% plot result on graph
figure('NumberTitle', 'off', 'Name', ['method2,data',num2str(dataType)])
plot( resultComMy);
grid on;
xlabel(['time of iteration x',num2str(deb)]);  
ylabel('loss function result');
legend(['  ridge regression loss function data',num2str(dataType)]);


figure('NumberTitle', 'off', 'Name', ['method2,data',num2str(dataType)])
plot(numToGoVector, errorRateMyTrain,'r',numToGoVector,errorRateMyTest,'g');
grid on;
xlabel('time of iteration');  
ylabel('errorRate');
legend('train','test');
ylim([0,1]);

% figure('NumberTitle', 'off', 'Name', ['method2,data',num2str(dataType)])
% plot( errorRateMyTrain);
% xlabel('time of iteration');  
% ylabel('errorRate');
% legend(' train');
% 
% figure('NumberTitle', 'off', 'Name', ['method2,data',num2str(dataType)])
% plot( errorRateMyTest);
% xlabel('time of iteration');  
% ylabel('errorRate');
% legend('test');
% 
% modelBuild = train(dataTrainLabel, sparse(dataTrainMatrix), '-s 6 -c 1 -B 1 -q');
% errorRateFromModel = predict_error_rate(modelBuild,dataTrainLabel,dataTrainMatrix);
% figure(3)
% plot( errorRateFromModel);
% legend(' matlabsparse logistic regression test error rate');

re = B;
end

