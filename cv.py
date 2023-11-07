import cv2
vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
while success:
  
  cv2.imwrite("frame.jpg", image)     # save frame as JPEG file      
  success,image = vidcap.read()
  cv2.imshow("frame",image)
  print('Read a new frame: ', success)
  count += 1