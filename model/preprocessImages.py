import cv2


def preprocessImages(inputImage):
    img = cv2.imread(inputImage)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img ,128, 255,cv2.THRESH_BINARY_INV)
    img = cv2.resize(th,(28,28) )
    return img
