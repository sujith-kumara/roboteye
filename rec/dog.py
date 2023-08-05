import cv2

# Load the image
img = cv2.imread('dataset/My_Dog.png')
# Load the pre-trained Haar Cascade classifier for dog detection
dog_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dogs = dog_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in dogs:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('Detected Dogs', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

