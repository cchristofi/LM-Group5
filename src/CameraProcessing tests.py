import robobo
import numpy as np
import math
import cv2

rob = robobo.SimulationRobobo().connect(port = 20000)

rob.play_simulation()
rob.set_phone_pan(pan_position = 1*math.pi, pan_speed = .5)
rob.set_phone_tilt(tilt_position = .55, tilt_speed = .1)

z = rob.get_image_front()
hsv = cv2.cvtColor(z, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (45, 70, 70), (85, 255, 255))


cv2.imshow("test_pictures",z)
cv2.waitKey(0)
cv2.imshow("test_pictures",mask)
cv2.waitKey(0)

#cv2.imwrite("test_pictures.png",image)



def convolve(data, kernel, overlap):
    kernel_h, kernel_w = kernel.shape
    stride_h = kernel_h - overlap
    stride_w = kernel_w - overlap
    height, width = data.shape
    
    n_h_c = (height - kernel_h) // stride_h + 1
    n_w_c = (width  - kernel_w)  // stride_w + 1
    
    result = np.zeros((n_h_c, n_w_c))
    
    for w in range(n_w_c):
        for h in range(n_h_c):
            X_slice = data[stride_h*h:(kernel_h +stride_h*h),stride_w*w:(kernel_w+stride_w*w)]
            result[h,w] = np.sum(np.multiply(X_slice, kernel))
    return result


data = mask
kernel = np.full((128, 16), 1/(128*16))
overlap = 0

import time
tic = time.time()
zz = convolve(mask, np.full((128, 16), 1/(128*16)), 0)
toc = time.time()
print(toc-tic)