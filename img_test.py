import cv2
import os
import time

path = "/media/aresx/DATA/yolo_v1_library/test/3/"  
files = os.listdir(path)

names = []

for img in files:
    #s = '/media/aresx/DATA/yolo_v1_library/test/3/'
    i = os.path.join(path, img)
    names.append(i)

for i in range(len(names)):
    img = cv2.imread(names[i])
    cv2.imshow('img_test', img)
    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'): #按q退出
        cv2.destroyAllWindows()