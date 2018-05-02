# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from pathlib2 import Path
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] Loading DNN network...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] Starting video stream...")

# open the stream of interest
if os.path.isfile('flight1.mp4'):
    cap = cv2.VideoCapture('flight1.mp4') # change to 0 for default webcam device
else:
    print ("[INFO] File not found.")
    print ("[INFO] Exiting program.")
    exit()

# give it some time to initialize
time.sleep(2)
if cap.isOpened(): 
    # Make sure video is open
    print ("[INFO] Capture stream opened.")

    # Get video dimensions for output streams
    width = cap.get(3)  # float
    height = cap.get(4) # float
    width = int(width)
    height = int(height)

    print ("[INFO] Video width: {0}").format(width)
    print ("[INFO] Video height: {0}").format(height)

    # Get video framerate
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps)

    print ("[INFO] Video framerate: {0}").format(fps)
else:
    exit("[INFO] Video stream could not be opened.")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Check if the file to write is already there
# delete if it is
try:
    os.remove('output.avi')
except OSError:
    pass

print ("[INFO] Beginning output stream...")
out = cv2.VideoWriter('output.avi',fourcc, fps, (width,height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the frame
        #out.write(frame)
        #cv2.imshow('frame',frame)

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
        cv2.putText(frame, "test!",(0, height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(225,255,255))
        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print ("[INFO] User requested exit.")
            break
    else:
        print ("[INFO] Capture stream has ended.")
        break

# Release everything if job is finished

print ("[INFO] Task complete.")
cap.release()
out.release()
cv2.destroyAllWindows()