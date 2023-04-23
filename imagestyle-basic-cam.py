import time
import cv2

w, h = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
net = cv2.dnn.readNetFromTorch("model/mosaic.t7")
time.sleep(2.0)

while True:
	_, frame = cap.read()
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)
	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release()
	


