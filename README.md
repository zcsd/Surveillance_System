# **Surveillance System**

## **Basic Function**

**1. Video Streaming(read from RTSP or usb webcam)**

**2. Motion Detection**

**3. Face Detection(based on HoG and CNN provided by Dlib)**

**4. Face Recognition(based on CNN provided by DLib, 99.38% accuracy in LFW test)**

**5. Face Images Collection**

**6. KNN/SVM Classifier Training(training on top of 128-D face encoding)**

**7. Update MySQL Server with event(NAME, TIMESTAMP, VIDEO_PATH)**

**8. Save Key Video Clip with motion to file (with Pre-Recording)**

## **Introduction**

This project is to develop a office/home/factory surveillance system, ip camera(access by RTSP) and usb webcam could be used as video stream source, a video clip will be saved if motion detected, it's done in surveillance.py; once there is a new saved video clip, video_analytics.py will start to read the video, do the face detection and recognition(multiple face), finally it will conclule who are in video, and send (name, timestamp, video_path) to SQL server.

Face detection and recognition depend on Dlib library, HoG is used in face detection because it's fast and accurate; Dlib return a 128-dimension face encoding for each face detected, on top of this 128D feature vector, I train a KNN/SVM classifier to recognize the person from my face database, the accuracy is very nice if the face quality is good. The face recognition model in DLib is a ResNet network with 27 conv layers, was trained from scratch on a dataset of about 3 million faces, achive 99.38% accuracy in LFW.

## **Usage**

**Directory structure for face database**
- Put face images into <SurveillanceSystem/faces/train> folder for next setp training, each person has one folder. You also can put test face images in <SurveillanceSystem/faces/test> if you want training accuracy.

```
    <SurveillanceSystem/faces/train>/
    ├── <person1>/
    │   ├── <somename1>.jpg
    │   ├── <somename2>.jpg
    │   ├── ...
    ├── <person2>/
    │   ├── <somename1>.jpg
    │   └── <somename2>.jpg
    └── ...
```

**Train a new classifier when face images databse changed:**
- python3 training.py

**Normal Usage(surveillance and face recognition):**
- ./start.sh (you need to "chmod +x start.sh" for first time using")

**Only use basic surveillance without face recognition:**
- python3 surveillance.py

**Press 'q' for quit program in GUI mode**

**NOTE:**
- You need to change some path variable and image ROI area for new setup PC and environemnt before using, detail in code comment. 

## **Installation**

**Requirements**
- Python3.5+
- Ubuntu(16.04/18.04)

**Steps:**

- Install Dlib(http://dlib.net/) with Python3 bindings

Install library dependancy(pass in Ubuntu 16.04/18.04):

```

sudo apt-get update

sudo apt-get install python3-pip build-essential cmake gfortran git wget curl graphicsmagick libgraphicsmagick1-dev libavcodec-dev libavformat-dev libgtk2.0-dev libjpeg-dev liblapack-dev libswscale-dev pkg-config python3-dev python3-numpy software-properties-common zip

sudo apt-get install libatlas-dev (change to libatlas-base-dev for Ubuntu 18.04)

sudo apt-get clean && rm -rf /tmp/* /var/tmp/*

pip3 install setuptools

```

Clone the code from github:

```

mkdir dlib; cd dlib

git clone https://github.com/davisking/dlib.git

```

Build the main dlib library:
```

mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1; cmake --build .

```

Build and install the Python3 extensions(NOT use CUDA):

```

cd ..

python3 setup.py install --yes USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA

```

At this point, you should be able to run ```python3``` and type ```import dlib``` successfully.

- Install face_recognition library(depend on Dlib):

```

pip3 install face_recognition

```

- Install necessary Python libary to import:

```

pip3 install opencv-python==3.4.2.17 (3.4.1 has bugs)

pip3 install pymysql

pip3 install scipy

pip3 install imutils

pip3 install sklearn

pip3 install numpy

sudo apt-get install python3-tk

pip3 install matplotlib

pip3 install mlxtend

pip3 install pyinotify

```

## **TODO**

- Reconnect SQL server if networt revover.

- Resend inforamtion in backup timelog to SQL server if network recover.

- Optimize multiple face recogniton in video-based analytics.

- Improve accuracy for unknown face recognition.

- Read config from file for easy setting.

- Be able to ignore light on/off event.

- Add web UI to dor daily view, access and control.

*Aug 31 2018*
