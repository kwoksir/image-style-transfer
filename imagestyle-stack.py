from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
from cvzone import *

m1 = "model/starry_night.t7"
m2 = "model/the_scream.t7"
m3 = "model/mosaic.t7"
m4 = "model/the_wave.t7"

net1 = cv2.dnn.readNetFromTorch(m1)
net2 = cv2.dnn.readNetFromTorch(m2)
net3 = cv2.dnn.readNetFromTorch(m3)
net4 = cv2.dnn.readNetFromTorch(m4)

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=320)
	orig = frame.copy()
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net1.setInput(blob)
	net2.setInput(blob)
	net3.setInput(blob)
	net4.setInput(blob)
	
	output1 = net1.forward()
	output1 = output1.reshape((3, output1.shape[2], output1.shape[3]))
	output1[0] += 103.939
	output1[1] += 116.779
	output1[2] += 123.680
	output1 /= 255.0
	output1 = output1.transpose(1, 2, 0)

	output2 = net2.forward()
	output2 = output2.reshape((3, output2.shape[2], output2.shape[3]))
	output2[0] += 103.939
	output2[1] += 116.779
	output2[2] += 123.680
	output2 /= 255.0
	output2 = output2.transpose(1, 2, 0)

	output3 = net3.forward()
	output3 = output3.reshape((3, output3.shape[2], output3.shape[3]))
	output3[0] += 103.939
	output3[1] += 116.779
	output3[2] += 123.680
	output3 /= 255.0
	output3 = output3.transpose(1, 2, 0)
	
	output4 = net4.forward()
	output4 = output4.reshape((3, output4.shape[2], output4.shape[3]))
	output4[0] += 103.939
	output4[1] += 116.779
	output4[2] += 123.680
	output4 /= 255.0
	output4 = output4.transpose(1, 2, 0)

	imgList = [output1, output2, output3, output4]
	imgStacked = stackImages(imgList,2,1)
	
	cv2.imshow("Combined", imgStacked)
	cv2.imshow("Orig", orig)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
	


