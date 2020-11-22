%% 测试pc端相机的帧率速度
cap = webcam('LRCP USB2.0 500W');
% resolutions = cap.AvailableResolutions();
%  '1280x800' ,fps:75 (default)
%  '1280x720' ,fps:80
%  '1024x768' ,fps:88
%  '800x600' ,fps:115
%  '640x400' ,fps:117
%  '352x288' ,fps:105
%  '320x240' ,fps:100
%  '160x120' ,fps:90
cap.Resolution = '1280x800';
img = cap.snapshot();
videoObj = vision.DeployableVideoPlayer();
step(videoObj,img);

t_start = tic;
FPS=0;
prev = FPS;
num = 0;
while videoObj.isOpen()
    num = num+1;
    img = cap.snapshot();

    eplapseT = toc(t_start);
    if abs(eplapseT-round(eplapseT))<0.01
        prev = FPS;
    else
        FPS = prev;
    end
    RGB = insertText(img,[10,20],FPS);
    step(videoObj,RGB);
    FPS = num/toc(t_start);
end
    







    