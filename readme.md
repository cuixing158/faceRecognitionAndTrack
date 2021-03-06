# Overview
This project implement the functions of matlab equivalent [MTCNN+ArcFace face recognition](https://github.com/cuixing158/jetson_faceTrack_pytorch )+tracking+HC08 Bluetooth module transmitting/receiving signal+voice broadcast, etc. The code implementation is relatively simple, condensed arduino control program+matlab algorithm design, and has a strong understanding.<br>

# Hardware and Software
- [usb camera](https://ww2.mathworks.cn/matlabcentral/fileexchange/45182-matlab-support-package-for-usb-webcams?s_tid=srchtitle )
- 2 servo
- [arduino](https://www.arduino.cc/ )
- [arduino support package](https://ww2.mathworks.cn/matlabcentral/fileexchange/47522-matlab-support-package-for-arduino-hardware?s_tid=srchtitle )
- blueTooth HC08
- matlab2020b+

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
3. Prepare face data, interact with `source/prepare_getFaces.m` on the pc, then execute `source/prepare_faceDatabase.m` to save database features, and finally run `source/camera_faceRec.m` <br>

# bluetooth control
![bluetooth](images/BLE.jpg)<br>

*Refer to [here](https://github.com/cuixing158/jetson_faceTrack_pytorch/blob/main/images/jetsonServo.jpg ) for the hardware connection diagram*<br>

# Reference
[Arduino-matlab](https://www.mathworks.com/hardware-support/arduino-matlab.html )<br>
[jetson_faceTrack_pytorch](https://github.com/cuixing158/jetson_faceTrack_pytorch )<br>
[Face Detection and Alignment MTCNN](https://github.com/matlab-deep-learning/mtcnn-face-detection )

# notes
author:cuixingxing <br>
email:cuixingxing150@gmail.com <br> 
