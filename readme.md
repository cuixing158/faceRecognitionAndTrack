# Overview
This project implement the functions of matlab equivalent [face detection/recognition](https://github.com/cuixing158/jetson_faceTrack_pytorch )+tracking+HC08 Bluetooth module transmitting/receiving signal+voice broadcast, etc. The code implementation is relatively simple, condensed arduino control program+matlab algorithm design, and has a strong understanding.<br>
实现了matlab等价的[人脸检测/识别](https://github.com/cuixing158/jetson_faceTrack_pytorch )+追踪+HC08蓝牙模块发射/接收信号+语音播报等功能，代码实现较为简洁，浓缩arduino控制程序+matlab算法设计，理解性强。

# Hardware and Software
- [usb camera](https://ww2.mathworks.cn/matlabcentral/fileexchange/45182-matlab-support-package-for-usb-webcams?s_tid=srchtitle )
- 2 servo
- [arduino](https://www.arduino.cc/ )
- [arduino support package](https://ww2.mathworks.cn/matlabcentral/fileexchange/47522-matlab-support-package-for-arduino-hardware?s_tid=srchtitle )
- blueTooth HC08
- >=matlab2020b
- [arduino support package](https://ww2.mathworks.cn/matlabcentral/fileexchange/47522-matlab-support-package-for-arduino-hardware?s_tid=srchtitle )

# Advantages
- Simple code, integrated control + algorithm.
- It is easy to debug the verification algorithm repeatedly.
- Fast development cycle.

# Disadvantages
- Compared with C++, the execution efficiency is lower
- Not easy to transplant
- SOTA algorithm unified integration into matlab is slow

# How to use
1. Download [arduino support package](https://ww2.mathworks.cn/matlabcentral/fileexchange/47522-matlab-support-package-for-arduino-hardware?s_tid=srchtitle );<br>
2. Configure the hardware environment, such as arduino wiring, steering gear wiring, jetson maximum power supply use short-circuit plug, the default is usb power supply, etc.;<br>
3. Prepare face data, interact with `prepare_getFaces.m` on the pc, then execute `prepare_faceDatabase.m` to save database features, and finally run `camera_faceRec.m` <br>

# bluetooth control
[bluetooth](images/BLE.jpg)<br>

*Refer to [here](https://github.com/cuixing158/jetson_faceTrack_pytorch/blob/main/images/jetsonServo.jpg ) for the hardware connection diagram*<br>

# Reference
[Arduino-matlab](https://www.mathworks.com/hardware-support/arduino-matlab.html )<br>
[jetson_faceTrack_pytorch](https://github.com/cuixing158/jetson_faceTrack_pytorch )<br>
[Face Detection and Alignment MTCNN](https://github.com/matlab-deep-learning/mtcnn-face-detection )

# notes
author:cuixingxing <br>
email:cuixingxing150@gmail.com <br> 
