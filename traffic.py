import os
import logging
import time
import logging.handlers
import random
#import matplotlib.pyplot as plt
import numpy as np
#import skvideo.io
import cv2
import imutils

from imutils.video import FileVideoStream

from imutils.video import FPS

import utils
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
random.seed(123)


from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)

# ============================================================================
IMAGE_DIR = "./out"
VIDEO_SOURCE = "../videos/1.mp4"
#VIDEO_SOURCE = "rtmp://185.55.24.19:1935/live/83-20181029062548637652.stream"
SHAPE = (720, 1280)  # HxW
#SHAPE = (576, 764)  # HxW

EXIT_PTS = np.array([
#     [[0, 720], [0, 620], [1000, 620], [1000, 720]],
#     #[[400, 764], [400, 400], [576, 400], [576, 764]],
      [[120, 720], [120, 590], [1280, 590], [1280, 720]],
#     #[[0, 350], [645, 350], [645, 0], [0, 0]]
])
# EXIT_PTS = np.array([
#     [[0,720], [0, 550], [50, 500], [1200,500], [1280, 550], [1280, 720]],

# ])
# ============================================================================

#def train_bg_subtractor(inst, cap, num=500):
def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    while True:
        im = cap.read()
        #print im.shape
        if not im.any():
            log.error("Frame capture failed, stopping...")
            continue

        #get height and width of current image         
        height, width, channels = im.shape

    #--------------------------Image Resizing-----------------------------------------------
        #if width > 900:
        newX,newY = 1280, 720
        im        = cv2.resize(im,(int(newX),int(newY)))

        inst.apply(im, None, 0.001)

        i += 1
        if i >= num:
            return cap


def main():
    log = logging.getLogger("main")

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # there is also bgslibrary, that seems to give better BG substruction, but
    # not tested it yet
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, detectShadows=True)

    # processing pipline for programming conviniance
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor, save_image=True, image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    # Set up image source
    # You can use also CV2, for some reason it not working for me

    #rtmp://185.55.24.19:1935/live/83-20181029062548637652.stream
    cap = FileVideoStream(VIDEO_SOURCE).start()

    #cap = skvideo.io.vreader(VIDEO_SOURCE)

    # skipping 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, cap, num=500)
    _frame_number = -1
    frame_number = -1

    #for frame in cap:
    while True:
        frame = cap.read()
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            continue

        #get height and width of current image         
        height, width, channels = frame.shape

    #--------------------------Image Resizing-----------------------------------------------
        #if width > 900:
        newX,newY = 1280, 720
        frame     = cv2.resize(frame,(int(newX),int(newY)))
        # print frame.shape

        # real frame number
        _frame_number += 1

        # skip every 2nd frame to speed up processing
        if _frame_number % 2 != 0:
            continue

        # frame number that will be passed to pipline
        # this needed to make video from cutted frames
        frame_number += 1


        # plt.show()
        # return

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        #cv2.imshow('frame', frame)
        
        pipeline.run()
        
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
# ============================================================================

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
