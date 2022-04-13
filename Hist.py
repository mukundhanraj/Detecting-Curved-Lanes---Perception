#Importing Libraries
import cv2
import numpy as np
import os

#Function to implement Histogram Equalization on a 2D array
def histo_eqliz(frm):
    histo = np.zeros(256)
    histo, bins = np.histogram(frm.flatten(), 256, [0, 255])
    histo = histo.cumsum()
    histo = histo / histo[-1]
    for x in range(frm.shape[0]):
        for y in range(frm.shape[1]):
            frm[x, y] = 255 * histo[frm[x, y]]
    return frm

#Function to implement Adaptive Histogram Equalization on a 2D array
def adpt_histo_eqliz(frm, k=30):

    adpt_img = np.zeros_like(frm)
    for i in range(0, frm.shape[0] - k, k):
        for j in range(0, frm.shape[1] - k, k):
            adpt_img[i:i+k, j:j+k] = histo_eqliz(frm[i:i+k, j:j+k])
    return adpt_img


if __name__ == '__main__':
    pt = 'suply/adaptive_hist_data'
    fls = os.listdir(pt)
    fls = sorted(fls)
    for f in fls:
        c_img = cv2.imread("suply/adaptive_hist_data/" + f)                 #Reading the images in a loop
        cv2.imshow('Input Set of Images', c_img)
        int_cr_cb = cv2.cvtColor(c_img, cv2.COLOR_BGR2YCR_CB)
        chnl = np.array(cv2.split(int_cr_cb))                               #Splitting Channels
        chnl[0] = histo_eqliz(chnl[0])
        int_cr_cb = cv2.merge(chnl)
        c_img = cv2.cvtColor(int_cr_cb, cv2.COLOR_YCR_CB2BGR)
        cv2.imshow('Output after Histogram Equalization', c_img)            #Acquiring output after histogram Equalisation
        chnl[0] = adpt_histo_eqliz(chnl[0])
        int_cr_cb = cv2.merge(chnl)                                         #Merging Channels
        c_img = cv2.cvtColor(int_cr_cb, cv2.COLOR_YCR_CB2BGR)
        cv2.imshow('Output after Adaptive Histogram Equalization', c_img)   #Acquiring output after adaptive histogram Equalisation
        cv2.waitKey(1)
