import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

def lbp(photo):
    def assign_bit(picture, x, y, c):  
        bit = 0
        try:
            if picture[x][y] >= c:
                bit = 1
        except:
            pass
        return bit

    def local_bin_val(picture, x, y):  
        eight_bit_binary = []
        centre = picture[x][y]
        powers = [1, 2, 4, 8, 16, 32, 64, 128]
        decimal_val = 0
        
        eight_bit_binary.append(assign_bit(picture, x-1, y + 1, centre))
        eight_bit_binary.append(assign_bit(picture, x, y + 1, centre))
        eight_bit_binary.append(assign_bit(picture, x + 1, y + 1, centre))
        eight_bit_binary.append(assign_bit(picture, x + 1, y, centre))
        eight_bit_binary.append(assign_bit(picture, x + 1, y-1, centre))
        eight_bit_binary.append(assign_bit(picture, x, y-1, centre))
        eight_bit_binary.append(assign_bit(picture, x-1, y-1, centre))
        eight_bit_binary.append(assign_bit(picture, x-1, y, centre))
        
        for i in range(len(eight_bit_binary)):
            decimal_val += eight_bit_binary[i] * powers[i]
        return decimal_val
    m, n, _ = photo.shape

    gray_scale = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    lbp_photo = np.zeros((m, n), np.uint8)

    for i in range(0, m):
        for j in range(0, n):
            lbp_photo[i, j] = local_bin_val(gray_scale, i, j)

    histogram = scipy.stats.itemfreq(lbp_photo)
    tmp=[]
    temp=""
    k=0
    j=0
    while k < 256:
            hist= str(histogram[j]).replace("[","").replace("]","").replace(" ","")
            if k < 10:
                if k != int(hist[0]):
                    temp = str(k) + "." + str(0)
                    k+=1
                    time.sleep(1)
                else:
                    temp = hist[0] + "."+hist[1:]
                    k+=1
                    j+=1
            elif k >= 10 and k < 100:
                if k != int(hist[0:2]):
                    temp = str(k) + "." + str(0)
                    k+=1
                else:            
                    temp = hist[0:2] + "." + hist[2:]     
                    k+=1 
                    j+=1
            elif k >= 100:
                if k != int(hist[0:3]):
                    temp = str(k) + "." + str(0)
                    k+=1
                else:
                    temp = hist[0:3] + "." + hist[3:]
                    k+=1
                    j+=1
            tmp.append(temp)
            temp = ""
    return lbp_photo, tmp


