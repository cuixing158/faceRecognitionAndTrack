function [predictName,predictScore] = classifyFace(inFeature,dataFeature)
% 功能：从数据库已有的人脸中找出最相似的人脸并返回人名和预测分数,即人脸识别
% 输入：
%     inFeature，1*512 single单张人脸图像的特征
%     dataFeature， 1*1结构体,含有person,feature域，形如，person: {'zhangsan','lisi','wangwu',...}，feature: {[1×512 single],[1×512 single],[1×512 single],...}
% 输出：
%     predictName， 1*1 string, 预测的人名
%     predictScore,1*1 single,预测的分数，[0,1]之间，值越大，可靠性越高
% author: cuixingxing
% date: 2020.11.14 
%
databankNames = dataFeature.person;
databankF = cat(1,dataFeature.feature{:});
scoresF = sum(inFeature.*databankF,2)./vecnorm(inFeature-databankF,2,2);
[predictScore,indMax] = max(scoresF);
predictScore = gather(predictScore);
predictName = databankNames{indMax};
end