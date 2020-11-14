function prepare_getFaces(webcamName,yourName)
% pc上在线获取并保存人脸图像，空格键保存，esc键或q键退出
%
% author:cuixingxing
% date:2020.11.14
%
arguments
    webcamName (1,1) string = "LRCP USB2.0 500W" % camera name
    yourName (1,1) string = "cuixingxing" % face name
end

%%
addpath('./mtcnn')
facedetector = mtcnn.Detector("MinSize", 15, "MaxSize", 70,'UseGPU',true);
scalar= 0.25; %图像缩放系数,small value speed fast detect

cap = webcam(webcamName);
cap.Resolution = '1280x800';
frame = snapshot(cap);
imshow(frame)
set(gcf,'CurrentCharacter','@'); % set to a dummy character
numFrame = 1;

%% capture face
isDone = false;
t1 = tic;
while(~isDone)
    frame = snapshot(cap);
    detectImg = imresize(frame,scalar);
    
    % 检测人脸
    [bboxes, scores, landmarks] = facedetector.detect(detectImg);
    if isempty(bboxes)
        imshow(frame);
        continue; 
    end
    [~,ind] = max(bboxes(:,3).*bboxes(:,4));
    bboxes(bboxes<0)=1;
    bbox = bboxes(ind,:)./scalar;
    score = scores(ind);
    landmark = landmarks(ind,:,:)./scalar;
    
    % draw
    RGB = insertObjectAnnotation(frame, "rectangle", bbox, score, "LineWidth", 2);
    RGB = insertMarker(RGB,squeeze(landmark),"square",...
        "Color","green");
    FPS = numFrame/toc(t1);
    RGB = insertText(RGB,[10,20],FPS);
    imshow(RGB)
    
    % 监听键盘按键相应事件,check for keys
    k=get(gcf,'CurrentCharacter');
    if k~='@' % has it changed from the dummy character?
        set(gcf,'CurrentCharacter','@'); % reset the character
        if k=='q'|| abs(k)==27 %  process the key( 'ESC or q') to exit
            close all
            clear('cap');
            isDone=true; 
        elseif abs(k) == 32 % space key, capture face image
            xcenter = bbox(1)+bbox(3)/2;
            ycenter = bbox(2)+bbox(4)/2;
            height = bbox(4);
            width = height;
            
            x = xcenter-width/2;
            y = ycenter-height/2;
            w = height;
            h = height;
            faceImg = imcrop(frame,[x,y,w,h]);
            saveFolder = fullfile("../dataSets/facebank",yourName);
            if ~exist(saveFolder,'dir')
                mkdir(saveFolder);
            end
            imgname = datestr(now,'yyyy_mm_dd_HH_MM_SS_FFF')+".jpg";
            imwrite(faceImg,fullfile(saveFolder,imgname))
            figure(gcf) % making a figure always on top
        end
    end
    numFrame = numFrame+1;
end

