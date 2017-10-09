# openCV
Open source computer vision

Ahoj, právě se učím s Githubem.
    Celkem mi to jde, jsem rád :).
    
Dalším krokem bude studování problematiky a vyhledávání informací o daném projektu.

-Domovská stránka OPENCV
http://opencv.org/

-YOUTUBE PLAYLIST TUTORIALS
https://www.youtube.com/watch?v=LvUcdfjYU78&list=PLSVkHwG9fSD67HB0uAYZ6GTRYcNgCN0Ti

https://www.youtube.com/watch?v=Y3ac5rFMNZ0&t=218s
https://github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp






-PYTHON TUTORIAL OPENCV 
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

-MAIN TUTORIAL OPENCV
https://hackernoon.com/tutorial-making-road-traffic-counting-app-based-on-computer-vision-and-opencv-166937911660

_________________________________________________________________________________________________________________
HARMONOGRAM PRÁCE:   
1) Nainstalování aplikace do VirtualBox
2) Propojení applikace s kamerou
3) Nahrání kodu
4) Vyladění kodu
5) Dokumentace k projektu
6) Vytvoření prezentace
_________________________________________________________________________________________________________________

INSTALACE OPENCV DO LINUXU:

-http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html


Hlavní myšlenka tohoto projektu je oddělení pozadí od pohybujících se objektů pomocí MOG(shadow detecting)


(KOD MOG)
import os
import logging
import logging.handlers
import random

import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt

import utils
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)

# ===========================================================
IMAGE_DIR = "./out"
VIDEO_SOURCE = "input.mp4"
SHAPE = (720, 1280)  # HxW
# ===========================================================


def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap:
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creting MOG bg subtractor with 500 frames in cache
    # and shadow detction
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = skvideo.io.vreader(VIDEO_SOURCE)

    # skipping 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)

    frame_number = -1
    for frame in cap:
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break

        frame_number += 1

        utils.save_frame(frame, "./out/frame_%04d.png" % frame_number)

        fg_mask = bg_subtractor.apply(frame, None, 0.001)
        
        utils.save_frame(frame, "./out/fg_mask_%04d.png" % frame_number)

# =========================================================

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

main()





















Instructions
____________________________________________________________________________-

1) Install OpenCV & get OpenCV source

 brew tap homebrew/science
 brew install --with-tbb opencv
 wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip
 unzip opencv-2.4.9.zip
 
2)Clone this repository

 git clone https://github.com/mrnugget/opencv-haar-classifier-training
 
3)Put your positive images in the ./positive_images folder and create a list of them:

 find ./positive_images -iname "*.jpg" > positives.txt
 
3)Put the negative images in the ./negative_images folder and create a list of them:

 find ./negative_images -iname "*.jpg" > negatives.txt
 
4)Create positive samples with the bin/createsamples.pl script and save them to the ./samples folder:

 perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
   "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
   -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 80 -h 40"
   
6)Use tools/mergevec.py to merge the samples in ./samples into one file:

 python ./tools/mergevec.py -v samples/ -o samples.vec
Note: If you get the error struct.error: unpack requires a string argument of length 12 then go into your samples directory and delete all files of length 0.

7)Start training the classifier with opencv_traincascade, which comes with OpenCV, and save the results to ./classifier:

 opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
   -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
   -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
   -precalcIdxBufSize 1024
If you want to train it faster, configure feature type option with LBP:

  opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
   -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
   -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
   -precalcIdxBufSize 1024 -featureType LBP
After starting the training program it will print back its parameters and then start training. Each stage will print out some analysis as it is trained:

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   1000 : 1000
NEG count : acceptanceRatio    600 : 1
Precalculation time: 11

+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|        1|
+----+---------+---------+
|   6|        1|        1|
+----+---------+---------+
|   7|        1| 0.711667|
+----+---------+---------+
|   8|        1|     0.54|
+----+---------+---------+
|   9|        1|    0.305|
+----+---------+---------+


END>
Training until now has taken 0 days 3 hours 19 minutes 16 seconds.
Each row represents a feature that is being trained and contains some output about its HitRatio and FalseAlarm ratio. If a training stage only selects a few features (e.g. N = 2) then its possible something is wrong with your training data.

At the end of each stage the classifier is saved to a file and the process can be stopped and restarted. This is useful if you are tweaking a machine/settings to optimize training speed.

8)Wait until the process is finished (which takes a long time — a couple of days probably, depending on the computer you have and how big your images are).

9)Use your finished classifier!

 cd ~/opencv-2.4.9/samples/c
 chmod +x build_all.sh
 ./build_all.sh
 ./facedetect --cascade="~/finished_classifier.xml"
 
 
Acknowledgements

A huge thanks goes to Naotoshi Seo, who wrote the mergevec.cpp and createsamples.cpp tools and released them under the MIT licencse. His notes on OpenCV Haar training were a huge help. Thank you, Naotoshi!


