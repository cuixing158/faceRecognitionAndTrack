function prepare_faceDatabase(onnxModelPath,networkInput,faceBankFolder)
% 功能：pc上在线获取人脸图像的特征并保存,供后期人脸识别比较使用。
% 输入：onnxModelPath,1*1 大小， string,人脸识别onnx模型文件路径，该onnx模型对一张112*112输入图像时输出是1*512维度的
%      networkInput，1*2 大小，形如[h,w]，指定网络输入高宽像素
%      faceBankFolder，1*1大小， string，指定的保存路径，保存每个人的特征向量和名字
%
% author:cuixingxing
% date:2020.11.14
%

arguments
   onnxModelPath (1,1) string ='../models/faceRecogPth/model_mobilefacenet.onnx';
    networkInput (1,2) double = [112,112];
    faceBankFolder (1,1) string = "../dataSets/facebank";
end
% 导入人脸识别模型
faceRecParams = importONNXFunction(onnxModelPath,'faceRecFcn');
 
%% 人脸数据库更新
imds = imageDatastore(faceBankFolder,...
    'IncludeSubfolders',true,...
    'FileExtensions',{'.jpg','.png'},...
    'LabelSource','foldernames');
nameClasses = categories(imds.Labels);
numsFaces = length(imds.Files);

imdsNew = transform(imds,@(x)preprocess(x,networkInput));
allFeatures = zeros(numsFaces,512,'single'); % 每行1*512代表一个人脸
num = 1;
while hasdata(imdsNew)
    faceImg = read(imdsNew);
    allFeatures(num,:) = faceRecFcn(faceImg,faceRecParams,...
        'Training',false,...
        'InputDataPermutation','none',...
        'OutputDataPermutation','none');
    num = num+1;
end

%% save per person feature
for i = 1:length(nameClasses)
    currentPerson = nameClasses(i);
    tempF = [];
    for j = 1:numsFaces
        if strcmp(string(imds.Labels(j)),currentPerson)
            tempF = [tempF;allFeatures(j,:)];
        end
    end
    st.person(i) = currentPerson;
    st.feature{i} = mean(tempF,1);
end
save(faceBankFolder+"/"+"facebank.mat","st")   
