function faceImg = preprocess(x,networkInput)
% 图像预处理，把matlab的h*w*c， RGB， [0,255]范围转换为n*c*h*w,RGB,[0,1]的输入图像
%
% author:cuixingxing
% 2020.11.14
%
faceImg = im2single(imresize(x,networkInput)); % [0,1] range
faceImg = permute(faceImg,[3,1,2]);% c*h*w
faceImg = reshape(faceImg,[1,3,networkInput]); % n*c*h*w，1*3*112*112, RGB, [0,1]输入
