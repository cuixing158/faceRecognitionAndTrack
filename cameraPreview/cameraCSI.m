%% 测试板载csi摄像头
jetsonObj = jetson('192.168.1.101','cuixingxing','cuixing1319746')
camlist = getCameraList(jetsonObj)

% show
cam = camera(jetsonObj,"vi-output, imx219 8-0010",[1280,720],'VideoDevice','/dev/video0')
img = snapshot(cam);
videoObj = vision.DeployableVideoPlayer();
step(videoObj,img);

t_start = tic;
num = 0;
while videoObj.isOpen()
    num = num+1;
    img = snapshot(cam);
    
    RGB = insertText(img,[10,20],num/toc(t_start));
    step(videoObj,RGB);
end