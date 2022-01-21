# Please install anaconda if having library issues https://www.anaconda.com/products/individual

# pip install Pillow
# pip install numpy
# pip install opencv-python (called cv2 once installed)
# pip install keyboard

""" 
if those dont work try this,
in my case from the video
conda activate SAI
for you 
conda activate [your env]

than do 

conda install opencv-python
"""
# or try this
# pipwin install opencv-python
# pipwin intall Pillow

# or try pip3 instead of pip if you still have issues

# General math on stuff
import numpy as np

# cv2 is our open-cv library
import cv2

# Pil or Pillow we are using for screenshots for the dino game
# there are LOTS OF LIBRARIES THAT CAN screen capture. This will just change the code for screenshots like I used in the dino game
# if this one doesnt work well..... look it up
from PIL import ImageGrab
from PIL import Image

# For emulating keystrokes for MAC it will be different 
# there are LOTS OF LIBRARIES THAT CAN EMULATE keystrokes. if this one doesnt work well..... look it up
import keyboard
import time


# My screen size is 1920 by 1080
# go to chrome://dino in chrome
def dino_game():
    zero = 0
    white = 255
    ahead = 0
    start = 200
    count = 0
    while True:
        img = ImageGrab.grab(bbox=(0, 360, 300,490)) #x, y, w, h
        img_np = np.array(img)
        frame = img_np
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        (thresh, frame) = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        # print(frame[100][175])
        # These frame coordinates go [y][x] starting from the top left
        if frame[84][175] != frame[129][175] or frame[90][175] != frame[129][175] or frame[100][175] != frame[129][175] or frame[100][180] != frame[129][175]:
            # keyboard.release("down")
            cv2.putText(frame,"White", (250,50), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
            keyboard.press("space")
            time.sleep(0.01)
            keyboard.release("space")
        else:
            cv2.putText(frame,"Black", (250,50), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
        print(time.time())
        count += int(time.time() - start)
        if count > start:
            count = 0 
            start = time.time()
            ahead += 1
        # Line coordinates go [x][y] starting top left
        cv2.line(frame,(170,90),(250,50),(255,5,0),1)
        # frame = cv2.line(frame, (170,84),(170,85),255,2)
        frame = cv2.line(frame, (170,90),(170,89),255,2)
        # frame = cv2.line(frame, (170,100),(170,101),255,2)
        # frame[120:124][100:120] = 255
        # print(pixel)
        # print(ImageGrab.grab(bbox=(0, 360, 300, 480)).size)
        # if img_np[92][ahead] == zero or img_np[92][ahead - 1] == zero or img_np[92][ahead - 2] == zero:
            # keyboard.release("down")
            # keyboard.press("space")
            # time.sleep(0.01)
            # keyboard.release("space")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0Xff == ord('q'):
            break
        
    cv2.destroyAllWindows()

def camera_feed():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while 1:
        ''' 
            cap.read is a function within the cv2 library that returns a tuple. we store the values from the 
            tuple, the first portion of the tuple as ret and the second as img 
            Even though we are storing it we are not using ret we only use the second value img
            another way to do this would be img = cap[1].read()
            (Which is a true or false value of if there is a frame or also called an image to capture)
        ''' 
        ret, img = cap[1].read()  
        cv2.imshow('img',img)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


""" 
For more info on HAAR cascades please follow this link https://www.pyimagesearch.com/2021/04/12/opencv-haar-cascades/

Haarcascades you can use have been installed under your cv2 library in pip or you can get them from here
    https://github.com/opencv/opencv/tree/master/data/haarcascades
"""
def camera_w_cascade():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print(cv2.data.haarcascades)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # this changes the resolution size of the camera
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while 1:
        # same thing as function camera_feed
        ret, img = cap.read()

        cv2.line(img,(320,0),(320,480),(255,5,0),1)
        cv2.line(img,(0,240),(640,240),(255,5,0),1)
        cv2.circle(img, (320, 240), 5, (255, 255, 255), -1)
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)

            arr = {y:y+h, x:x+w}
            print (arr)
            
            print ('X :' +str(x))
            print ('Y :'+str(y))
            print ('x+w :' +str(x+w))
            print ('y+h :' +str(y+h))

            xx = int((x+(x+h))/2)
            yy = int((y+(y+w))/2)

            print (xx)
            print (yy)

            center = (xx,yy)

            print("Center of Rectangle is :", center)
            data = "X{}Y{}".format(int(xx/2.8), int(yy/2.8))
            print ("output = '" +data+ "'")
            # arduino.write(data.encode())
        

        cv2.imshow('img',img)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
def blob():
# (hMin = 46 , sMin = 107, vMin = 90), (hMax = 110 , sMax = 222, vMax = 252)
  img = cv2.imread("D:\Coding\AIsociety\OpenCV\\circles.png", cv2.IMREAD_COLOR)
  hMin = 46
  sMin = 107
  vMin = 90

  hMax = 110
  sMax = 222
  vMax = 252

  # Set minimum and max HSV values to display
  lower = np.array([hMin, sMin, vMin])
  upper = np.array([hMax, sMax, vMax])
  canvas = img.copy()
  # Create HSV Image and threshold into a range.
  hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower, upper)
  mask = cv2.bitwise_not(mask)
  # blur = cv2.blur(mask, (5,5), 0)
  cv2.imshow("Mask", mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  params = cv2.SimpleBlobDetector_Params()

#   params.minThreshold = 10
#   params.maxThreshold = 200

# #   Filter by Area.
#   params.filterByArea = True
#   params.minArea = 10

#   # Filter by Circularity
#   params.filterByCircularity = False
#   params.minCircularity = 0.1

#   # Filter by Convexity
#   params.filterByConvexity = False
#   params.minConvexity = 0.87

#   # Filter by Inertia
#   params.filterByInertia = False
#   params.minInertiaRatio = 0.01
  detector = cv2.SimpleBlobDetector_create(params)

  keypoints = detector.detect(mask)
  
  keys = cv2.drawKeypoints(img, keypoints, np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imshow("Keys", keys)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return keypoints

def contour_pic():
    # image = cv2.imread('D:\Coding\AIsociety\OpenCV\circles.png')
    image = cv2.imread('D:\Coding\AIsociety\OpenCV\\rose.jpg')
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    # cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()

    for i in contours:
        # area = cv2.contourArea(i)
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # see the results
    cv2.imshow('Contours', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_cam():
    cap = cv2.VideoCapture(0)
    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)

        contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        for i in contours:
            area = cv2.contourArea(i)
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# dino_game()
# blob()
# contour_cam()