function camera_faceRec(arduinoObj,BLEsserial,audioFolder,faceOnnxFile,...
    networkInput,faceBankFile,cameraName,cameraResolution,...
    servo1Port,servo2Port,...
    rangeHcam,rangeVcam,rangeHservo,rangeVservo,scalar,options)
% 功能：云台人脸检测识别跟踪+HC08蓝牙模块发射/接收控制信号+语音播报
%
% algorithm:MTCNN+ArcFace(mobilenet backbone)
%
% author:cuixingxing
% 2020.9.2 Create this file
% 2020.11.14 modify, add face recognition, arguments
% performance: Matlab2020b+GTX1070+win10+i7-6700 , faceDetect ~10FPS,
% faceDetect+ faceRecognition ~5FPS
%
arguments
    arduinoObj (1,1)  arduino = arduino("COM4","Mega2560","Libraries",{'Serial','Servo'}) % usb有限连接，Serial蓝牙模块，Servo舵机模块
    BLEsserial (1,1)  = device(arduinoObj,'SerialPort',3,'BaudRate', 9600) %  TxPin: 'D14' RxPin: 'D15'
    audioFolder  {mustBeFolder} = '../dataSets' % audio files
    faceOnnxFile {mustBeFile} = '../models/faceRecogPth/model_mobilefacenet.onnx' % face recognition model file
    networkInput (1,2) {mustBeNumeric} = [112,112]
    faceBankFile {mustBeFile} = "../dataSets/facebank/facebank.mat"
    cameraName {mustBeA(cameraName,{'single','double','char'})} = 'LRCP USB2.0 500W'
    cameraResolution {mustBeText} = '1280x800'
    servo1Port {mustBeText} = 'D52' % 云台下面舵机与arduino数字接口，左右摆动
    servo2Port {mustBeText} = 'D53' % 云台上面舵机与arduino数字接口，上下摆动
    rangeHcam (1,1) {mustBePositive} = 18 % 摄像头水平方向视角范围（单位：度）, 也可以理解为调节灵敏度
    rangeVcam (1,1) {mustBePositive} = 10 % 摄像头竖直方向视角范围（单位：度）, 也可以理解为调节灵敏度
    rangeHservo (1,1) {mustBePositive} = 180 % 舵机水平方向范围（单位：度）
    rangeVservo (1,1) {mustBePositive} = 180 % 舵机竖直方向范围（单位：度）
    scalar (1,1) double {mustBePositive} = 0.5 %图像缩放系数,系数越小，越有利于加快速度，但会漏检
    options.useFaceRec (1,1) string {mustBeMember(options.useFaceRec,["MobilenetFace","SoftMaxFace","ArcFace","None"])} = "ArcFace" % Name-Value pair
end

%% import all files
addpath('./mtcnn') 
addpath('./utils') 

% audioFiles = dir(fullfile(audioFolder,'*.m4a'));
[y1,Fs1] = audioread(fullfile(audioFolder,'faceDetect.m4a'));
player = audioplayer(y1,Fs1);

% import Arc faceRecognition model
faceRecModel = options.useFaceRec;
if strcmpi(faceRecModel,"MobilenetFace")
    faceRecParams = importONNXFunction(faceOnnxFile,'faceRecFcn');
elseif strcmpi(faceRecModel,"SoftMaxFace")
    load('../models/faceRecogPth/mobilenetv2_softmax.mat','net');
elseif  strcmpi(faceRecModel,"ArcFace")
    load('../models/faceRecogPth/mobilenetv2_arcface.mat','net');
end

predictName = "unknow";
predictScore = -1;
load(faceBankFile,'st') % 人脸数据库特征，每个人对应1*512特征
if ~strcmpi(faceRecModel,"None") && all(isfield(st,{'person','feature'}))
    fprintf("The face database has person:"+join(repmat("%s",1,length(st.person)))+"\n",st.person{:});
else
    faceRecModel = false;
    warning("The face database have no person,please run prepare_getFaces.m and prepare_faceDatabase.m respectively! or 'useFaceRec' is set to false")
end

% servo init
s1 = servo(arduinoObj,servo1Port); 
s2 = servo(arduinoObj,servo2Port); 
degree_x = 0.8;% 108/180; % [0,1]范围内初始化相机角度
degree_y = 0.3;% 54/180; % [0,1]范围内初始化相机角度
writePosition(s1,degree_x);% 初始化角度位置
writePosition(s2,degree_y);% 初始化角度位置

%% set parameters
cap = webcam(cameraName);
cap.Resolution = cameraResolution;
% preview(cap);
ratioHorizontal = rangeHcam/rangeHservo; % 相机与舵机水平方向转动比例
ratioVertical = rangeVcam/rangeVservo; % 相机与舵机竖直方向转动比例
facedetector = mtcnn.Detector("MinSize", 20, "MaxSize", 100,'UseGPU',true);

%% loop
frame = snapshot(cap);
videoObj = vision.DeployableVideoPlayer();
step(videoObj,frame)
[H,W,~] = size(frame);
imgCenter = [W/2,H/2]; % 图像固有中心坐标，如[640/2,480/2]
numFrame = 0;
isPause = false;% 是否进行人脸检测+跟踪+识别+语音
t1 = tic;
while(videoObj.isOpen())
    % 接收蓝牙信号，手机发送stop停止人脸识别，发送start开始人脸识别,发送pause暂停
    numBytes = BLEsserial.NumBytesAvailable;% NumBytesAvailable可理解为HC08蓝牙模块的缓存数据大小
    if numBytes>0
        readdata = char(read(BLEsserial,numBytes)); % numBytes>0，才可以读入，并且读的数据为ascii码值，转换为char供可读
        fprintf('read data from mobile:%s\n',readdata);
        if strcmp(readdata,'stop') % 手机发送stop跳出循环
            break;
        elseif strcmp(readdata,'start')
            isPause =false;
        elseif strcmp(readdata,'pause')
            isPause =true;
        end
    end
    
    if ~isPause
        % start
        numFrame = numFrame+1;
        writePosition(s1,degree_x);
        writePosition(s2,degree_y);
%         fprintf("degree_x:%.2f,degree_y:%.2f\n",degree_x*180,degree_y*180);
        frame = snapshot(cap);
        detectImg = imresize(frame,scalar);
        
        % 检测人脸
        [bboxes, scores, landmarks] = facedetector.detect(detectImg);
        if isempty(bboxes)
            step(videoObj,frame);
            write(BLEsserial,"no face ");
            continue;
        end
        [~,ind] = max(bboxes(:,3).*bboxes(:,4));
        bboxes(bboxes<0)=1;
        bbox = bboxes(ind,:)./scalar;
        score = scores(ind);
        landmark = landmarks(ind,:,:)./scalar;
        
        % 人脸识别
        if ~strcmpi(faceRecModel,"None")
            xcenter = bbox(1)+bbox(3)/2;
            ycenter = bbox(2)+bbox(4)/2;
            height = bbox(4);
            width = height;
            
            x = xcenter-width/2;
            y = ycenter-height/2;
            w = height;
            h = height;
            faceImg = imcrop(frame,[x,y,w,h]);
            if canUseGPU()
                if strcmpi(faceRecModel,"MobilenetFace") % onnx model
                    faceImg = gpuArray(preprocess(faceImg,networkInput)); % n*c*h*w，1*3*112*112, RGB, [0,1]输入
                    outFeature = faceRecFcn(faceImg,faceRecParams,...
                        'Training',false,...
                        'InputDataPermutation','none',...
                        'OutputDataPermutation','none');
                    [predictName,predictScore] = classifyFace(outFeature,st);
                elseif strcmpi(faceRecModel,"SoftMaxFace") || strcmpi(faceRecModel,"ArcFace") % matlab model
                    faceImg = gpuArray(imresize(faceImg,networkInput));
                    [predictName,predictScore] = classify(net,faceImg);
                    predictScore = max(predictScore);
                end
            end
        end
        
        % 发送蓝牙信号
        write(BLEsserial,"Name:"+string(predictName)+",Score:"+string(predictScore)+"  ");
        
        % 语音播报
        if strncmp(string(predictName),"zhangsan",7)
            play(player); % Play without blocking，无阻塞方式播报
        end
        
        %绘图
        RGB = insertObjectAnnotation(frame, "rectangle", bbox, score, "LineWidth", 2);
        RGB = insertMarker(RGB,squeeze(landmark),"square",...
            "Color","green");
        display = sprintf("FPS:%.2f,%s,score:%.2f",numFrame/toc(t1),predictName,predictScore);
        RGB = insertText(RGB,[10,20],display,'FontSize',30);
        step(videoObj,RGB);
        
        %% 调整角度，假设相机无畸变
        center = [bbox(1)+bbox(3)/2,bbox(2)+bbox(4)/2]; % [x,y]
        if center(1)<=imgCenter(1) % 人脸在图像左边
            distance_x = imgCenter(1)-center(1);
            degree_x = readPosition(s1)+ (distance_x./W).*ratioHorizontal;
        else
            distance_x = center(1)-imgCenter(1);
            degree_x = readPosition(s1)-(distance_x./W).*ratioHorizontal;
        end
        if center(2)<=imgCenter(2) % 人脸在图像上方
            distance_y = imgCenter(2)-center(2);
            degree_y = readPosition(s2)+(distance_y/H).*ratioVertical;
        else
            distance_y = center(2)-imgCenter(2);
            degree_y = readPosition(s2)-(distance_y/H).*ratioVertical;
        end
        
        %% 限定范围内
        degree_x(degree_x<0)=0;
        degree_y(degree_y<0)=0;
        degree_x(degree_x>1)=1;
        degree_y(degree_y>1)=1;
    end
end
end


