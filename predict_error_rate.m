function [ errorRate ] = predict_error_rate( B, data1TestLabel, data1TestMatrix  )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[m,~] = size(data1TestLabel);
remp = data1TestMatrix*(B');
D = ones(size(remp));
D  ((remp<0)) = -1;
T = D  + data1TestLabel;
errorNum = numel(find(T == 0) );
errorRate = errorNum/m;
end

