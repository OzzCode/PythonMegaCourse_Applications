import cv2

face_cascade = cv2.CascadeClassifier('resource_files/haarcascade_frontalface_default.xml')

img = cv2.imread('resource_files/ajw_recognition2.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the face of image using pixel dimensions
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                      minNeighbors=5)

# create a rectangle around image to detect in pixels
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Prints type of faces object
print(type(faces))  # Numpy Array
print(faces)

resized = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
cv2.imwrite('resource_files/face_resized.jpg', resized)

# cv2.imshow('Gray_Image', gray_img)
# cv2.imshow('Gray_Image', img)
cv2.imshow('Gray_Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
