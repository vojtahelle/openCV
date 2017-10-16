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



https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Cpp



SMILE DETECTION

https://aminesehili.wordpress.com/2015/09/20/smile-detection-with-opencv-the-nose-trick/
https://github.com/amsehili/fnsmile


import sys
import os
import cv2

from optparse import OptionParser

__all__ = []
__version__ = 0.1
__date__ = '2015-02-23'
__updated__ = '2015-09-19'

DEBUG = 0
TESTRUN = 0
PROFILE = 0


class HaarObjectTracker():
    
    def __init__(self, cascadeFile, childTracker=None, color=(0,255,0), id="", verbose=0):
        
        self.detector = cv2.CascadeClassifier(cascadeFile)
        self.childTracker = childTracker
        self.color = color
        self.id = id
        self.verbose = verbose
        
        
    def detectAndDraw(self, frame, drawFrame):
        
        if self.childTracker is not None:
            childDetections = self.childTracker.detectAndDraw(frame, drawFrame)
        
        else:
            childDetections = [(frame, 0, 0, frame.shape[1], frame.shape[0])]
         
        retDetections = []
           
        for (pframe, px, py, pw, ph) in childDetections:
            
            ownDetections = self.detector.detectMultiScale(pframe, scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(30, 30),
                                                     flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            
            
            
            for (x, y, w, h) in ownDetections:
                cv2.rectangle(drawFrame, (x+px, y+py), (x+px+w, y+py+h), self.color, 2)
                rframe = pframe[y:y+h, x:x+w]
                # translate coordinate to correspond the the initial frame
                retDetections.append((rframe, x+px, y+py, w, h))

        self.trace(len(retDetections))
        
        return retDetections
    
    
    
    def trace(self, nbrObjects):
        if self.id != "" and nbrObjects > 0 and self.verbose > 0:
            print("{id}: {n} detection(s)".format(id=self.id, n=nbrObjects))
        
class LowerFaceTracker(HaarObjectTracker):
    
    def __init__(self, noseCascadeFile, childTracker=None, color=(0,0,255), id=""):
        HaarObjectTracker.__init__(self, noseCascadeFile, childTracker, color, id)

    
    def detectAndDraw(self, frame, drawFrame):
        
        if self.childTracker is not None:
            
            childDetections = self.childTracker.detectAndDraw(frame, drawFrame)
        
        else:
            childDetections = [(frame, 0, 0, frame.shape[1], frame.shape[0])]
         
        retDetections = []
           
        for (pframe, px, py, pw, ph) in childDetections:
            
            noses = self.detector.detectMultiScale(pframe, scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(30, 30),
                                                     flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
            
            # select the lowest nose
            lowest_y = 999999L
            for (nx, ny, nw, nh) in noses:
                if ny < lowest_y:
                    lowest_y = ny
                    lx = ny
                    lw = nw
                    lh = nh

            if lowest_y == 999999L: # no nose detected
                continue
            
            
            # draw an ellipse around the lowest detected nose
            cv2.ellipse(drawFrame, (px+lx , py+lowest_y + int(lh/2)) , (lw//2,lh//2), 0, 0, 360, self.color, 2)
            
            # extract the lowest part of the face starting from the center of the nose
            center_y = lowest_y + int(float(lh) / 2)
            lowerFrame = pframe[center_y:,0:]
            
            # translate coordinates to correspond to the the initial frame
            retDetections.append((lowerFrame, px, py + center_y, lowerFrame.shape[1], lowerFrame.shape[0]))
            
            # lower face's rectangle
            cv2.rectangle(drawFrame, (px + 2, py + center_y), (px + pw - 2, py + ph - 2), (255, 255, 255), 2)
            
        self.trace(len(retDetections))
            
        return retDetections
    
    


def main(argv=None):
    '''Command line options.'''
    
    program_name = os.path.basename(sys.argv[0])
    program_version = "v0.1"
    program_build_date = "%s" % __updated__
 
    program_version_string = '%%prog %s (%s)*****' % (program_version, program_build_date)
    program_usage = "Usage: python %s -s smile_model [-f face_model [-n nose_model]] [-d video_device].\
    \nFor more information run python %s --help" % (program_name, program_name)
    
    program_longdesc = "Read a video stream from a capture device or of file and track smiles on each frame.\
                        The default behavior is to scan the whole frame in search of smiles.\
                        If a face model is supplied (option -f), then search a smile is searched on the detected face.\
                        If a nose model is supplied (option -n, requires -f), the a smile is searched within the lower\
                        part of the face, taking the nose as a split point."
                        
    program_license = "Copyright 2015 Amine Sehili (<amine.sehili@gmail.com>).  Licensed under the GNU CPL v03"
 
    if argv is None:
        argv = sys.argv[1:]
    try:
        # setup option parser
        parser = OptionParser(version=program_version_string, epilog=program_longdesc, description=program_longdesc)
        parser.add_option("-f", "--face-model", dest="face_model", default=None, help="Cascade model for face detection (optional)", metavar="FILE")
        parser.add_option("-n", "--nose-model", dest="nose_model", default=None, help="Cascade model for nose detection (optional, requires -f)", metavar="FILE")
        parser.add_option("-s", "--smile-model", dest="smile_model", default=None, help="Cascade model for smile detection (mandatory)", metavar="FILE")
        
        parser.add_option("-v", "--video-source", dest="video_source", default="0", help="If an integer, try reading from the webcam, else open video file", metavar="INT/FILE")
        
    
        (opts, args) = parser.parse_args(argv)
        
        if opts.smile_model is None:
            raise Exception(program_usage)
        
        if opts.nose_model is not None and opts.face_model is None:
            raise Exception("Error: a nose detector also requires a face model")
        
        
        if opts.face_model is not None:
            faceTracker = HaarObjectTracker(cascadeFile=opts.face_model, id="face-tracker")
            
            if opts.nose_model is not None:
                lowerFaceTracker = LowerFaceTracker(noseCascadeFile = opts.nose_model, childTracker = faceTracker, id="nose-tracker")
                smileTracker = HaarObjectTracker(opts.smile_model, lowerFaceTracker, (255,0,00), "smile-tracker", verbose=1)
            else:
                smileTracker = HaarObjectTracker(opts.smile_model, faceTracker, (255,0,0), "smile-tracker", verbose=1)
            
        else:
            smileTracker = HaarObjectTracker(opts.smile_model, None, (255,0,0), "smile-tracker", verbose=1)
            
        
        try:
            videoCapture = cv2.VideoCapture(int(opts.video_source))
            videoCapture.grab()
        except ValueError as e:
            videoCapture = cv2.VideoCapture(opts.video_source)
        
        
        while True:
            
            videoCapture
            ret, frame = videoCapture.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            smileTracker.detectAndDraw(gray, frame)
            
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
        videoCapture.release()
        cv2.destroyAllWindows()        
            
    
    except KeyboardInterrupt as k:
        sys.stderr.write("program will exit\nBye!\n")
        return 0
       
    except Exception, e:
        sys.stderr.write(str(e) + "\n")
        return 2


if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = '_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())







https://www.youtube.com/watch?v=hN4lULtjzzE
https://github.com/cmusatyalab/openface

https://www.youtube.com/watch?v=mXfQTBclLqE



https://github.com/cmusatyalab/openface/tree/master/openface



Commands For installing PIP

1 $ sudo apt-get install python-setuptools python-dev build-  essential
2 $ sudo easy_install pip
3 $ sudo pip install --upgrade virtualenv

To Install Packages via PIP

1 $ sudo pip install numpy
2 $ sudo pip install cython
3 $ sudo pip install scipy
4 $ sudo pip install scikit-learn
5 $ sudo pip install scikit-image
6 $ sudo pip install pandas




$ sudo apt-get install build-essential
$ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

changing the directory and downloading opencv

$ cd /usr/src
$ sudo git clone https://github.com/opencv/opencv.git
$ cd ~/opencv
$ sudo mkdir release
$ cd release
$ sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
$ sudo make
$ sudo make install

$ pkg-config --modversion opencv





$ mkdir -p ~/src
$ cd ~/src
$ tar xf dlib-18.16.tar.bz2
$ cd dlib-18.16/python_examples
$ mkdir build

!--before proceeding further chk for the boost error--!
For Boost Installation
to fix error boost error
$ wget -O boost_1_62_0.tar.gz http://sourceforge.net/projects/boost...
$ tar xzvf boost_1_62_0.tar.gz
$ cd boost_1_62_0/
$ sudo apt-get update
$ sudo apt-get install build-essential g++ python-dev autotools-dev libicu-dev build-essential libbz2-dev libboost-all-dev


$ ./bootstrap.sh --prefix=/usr/local

Then build it with:

$ ./b2
and eventually install it:

$ sudo ./b2 install

!--Now Continue Further--!

After boost, Accessing the directory 

$ cd ~/src
$ cd dlib-18.16/python_examples
$ cd build

now the leftovers commands

$ cmake ../../tools/python
$ cmake --build . --config Release
$ sudo cp dlib.so /usr/local/lib/python2.7/dist-packages


$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch
$ bash install-deps
$ ./install.sh

$ sudo apt-get install luarocks
TORCH_LUA_VERSION=LUA52 ./install.sh

--For permission issues or errors with luarocks type the following commands--

$ sudo chmod -R 777 ~/opencv
$ sudo chmod -R 777 ~/torch

--For permission issues --

packages to download via luarocks for Torch

$ luarocks install dpnn 
$ luarocks install nn 
$ luarocks install optim 
$ luarocks install csvigo 
$ luarocks install cutorch and cunn (only with CUDA) 
$ luarocks install fblualib (only for training a DNN) 
$ luarocks install tds (only for training a DNN) 
$ luarocks install torchx (only for training a DNN) 
$ luarocks install optnet (optional, only for training a DNN)
