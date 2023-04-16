import cv2

image = cv2.imread('tute.jpg')
(H, W) = image.shape[:2]

#net = cv2.dnn.readNetFromTorch('model\eccv16\starry_night.t7')
net = cv2.dnn.readNetFromTorch('model\mosaic.t7')
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (0, 0, 0), swapRB=False, crop=False)

net.setInput(blob)
out = net.forward()
out = out.reshape(out.shape[1], out.shape[2], out.shape[3])

cv2.normalize(out, out,alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
result = out.transpose(1, 2, 0)
cv2.imshow('original', image)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()
