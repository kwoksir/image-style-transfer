from imutils.video import VideoStream
from imutils import paths
import itertools
import imutils
import time
import cv2

modelPaths = paths.list_files("model", validExts=(".t7",))
modelPaths = sorted(list(modelPaths))
models = list(zip(range(0, len(modelPaths)), (modelPaths)))
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)
net = cv2.dnn.readNetFromTorch(modelPath)

while True:
    image = cv2.imread('sample.png')
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    output = output.reshape(output.shape[1], output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
  
    output /= 255.0
    
    #cv2.normalize(output, output,alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    output = output.transpose(1, 2, 0)
    
    cv2.imshow('original', image)
    cv2.imshow('result', output)
    cv2.setWindowTitle('result', str(modelPath[6:-3]))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        (modelID, modelPath) = next(modelIter)
        net = cv2.dnn.readNetFromTorch(modelPath)
    elif key == ord("q"):
        break
cv2.destroyAllWindows()
