# All rights reserved under IndiEND^tm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import confusion_matrix
import csv
import pickle


def otsu(im):
    im_flat = np.reshape(im, (im.shape[0] * im.shape[1]))

    [hist, _] = np.histogram(im, bins=256, range=(0, 255))
    # Normalization so we have probabilities-like values (sum=1)
    hist = 1.0 * hist / np.sum(hist)

    val_max = -999
    thr = -1
    for t in range(1, 255):
        # Non-efficient implementation
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        m1 = np.sum(np.array([i for i in range(t)]) * hist[:t]) / q1
        m2 = np.sum(np.array([i for i in range(t, 256)]) * hist[t:]) / q2
        val = q1 * (1 - q1) * np.power(m1 - m2, 2)
        if val_max < val:
            val_max = val
            thr = t

    new_im = np.zeros(im.shape, dtype=int)
    new_im[im > thr] = 255
    print(type(new_im))
    return new_im, thr


def PreProcc(im_color, enchancement=True, segmentation='Edim Thresholding', th=0.6, edge_filter='Adaptive STD', kernel_type="Gaussian", contour_font=None, Filter_sz=7, adaptive_filter_size=25):
    im_color_16 = im_color.copy().astype('float64')
    im = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)

    if enchancement == True:
        # ENCHANCE

        im_hsv = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)
        im_V = np.float64(im_hsv[:, :, 2])

        mean = np.mean(im_V)
        mean_sqr = np.mean(np.square(im_V))
        std = np.sqrt(mean_sqr - np.square(mean))
        one = np.ones(im_V.shape)

        Sk = np.mean(np.power(im_V - mean * one, 3)) / (np.power(std, 3))

        a = 220
        b = 2.5

        w = (b / a) * np.multiply(np.power(im_V / a, (b - 1)),
                                  np.exp(-np.power(im_V / a, b)))

        mean_v = np.mean(w)
        new_V = w.copy()
        if Sk > 0:
            new_V[w < mean_v] = mean_v
        else:
            new_V[w > mean_v] = mean_v

        new_V = 255 * new_V / np.max(new_V)
        ench_img = im_hsv.copy()
        ench_img[:, :, 2] = new_V
        im_color_16 = cv2.cvtColor(ench_img, cv2.COLOR_HSV2BGR)

    # CIVE thresholding
    if segmentation == 'CIVE':
        cive_im = 0.441 * im_color_16[:, :, 2] - 0.811 * im_color_16[:, :, 1] + 0.385 * im_color_16[:, :,
                                                                                                    0] + 18.78745 * np.ones(
            im.shape)

        mcive_im = (cive_im - np.ones(cive_im.shape) *
                    np.min(cive_im)) / (np.max(cive_im) - np.min(cive_im))
        new_im = np.zeros(im.shape)
        indx = (mcive_im < th)
        new_im[indx] = 255

    # Edim Thresholding
    elif segmentation == 'Edim Thresholding':
        cive_im = 0.441 * im_color_16[:, :, 2] - 0.811 * im_color_16[:, :, 1] + 0.385 * im_color_16[:, :,
                                                                                                    0] + 18.78745 * np.ones(
            im.shape)
        mcive_im = (cive_im + np.ones(cive_im.shape) * 188.0176) / 417.4351
        ngrdi_im = np.true_divide((im_color_16[:, :, 1] - im_color_16[:, :, 2]),
                                  (im_color_16[:, :, 1] + im_color_16[:, :, 2]))
        ngrdi_im[ngrdi_im == np.inf] = 0

        (ngrdi_im-np.ones(ngrdi_im.shape)*np.nanmin(ngrdi_im)) / \
            (np.nanmax(ngrdi_im)-np.nanmin(ngrdi_im))

        hsv = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)

        im_V = np.float64(hsv[:, :, 2])
        kv = 0.0001
        kg = 0.1

        ngrdi_im = np.true_divide((im_color_16[:, :, 1] - im_color_16[:, :, 2]),
                                  (im_color_16[:, :, 1] + im_color_16[:, :, 2]))
        ngrdi_im[ngrdi_im == np.inf] = 0

        mngrdi_im = (ngrdi_im - np.ones(ngrdi_im.shape) * np.nanmin(ngrdi_im)) / (
            np.nanmax(ngrdi_im) - np.nanmin(ngrdi_im))

        hybrid = kv * (1 - np.sqrt(im_V / 255)) * im_color[:, :, 2] + kg * 255 * mngrdi_im + 255 * kv * (
            im_V) * np.absolute(hsv[:, :, 0] - 60) / 120 + 255 * (1 - kv - kg) * (1 - mcive_im)

        indx = hybrid > 255 * th
        new_im = np.zeros(im.shape)
        new_im[indx] = 255
    else:
        print("Error! No such segmentation type")
        return None, None, None

    # Dilation + Erosion

    if kernel_type == "Gaussian":
        kernel = np.outer(cv2.getGaussianKernel(Filter_sz, 1),
                          cv2.getGaussianKernel(Filter_sz, 1))
    else:
        kernel = np.ones((Filter_sz, Filter_sz), np.uint8)

    # img_dilation = cv2.dilate(new_im, kernel, iterations=3)
    # img_diler = cv2.erode(img_dilation, kernel, iterations=3)

    pro_im = new_im
    for i in range(5):
        pro_im = cv2.erode(pro_im, kernel, iterations=1)
        pro_im = cv2.dilate(pro_im, kernel, iterations=1)

    # Inverse Thresholding
    inv_th_im = im_color.copy()  # cv2.imread("drive/My Drive/EE493/Ripeness/input.jpg")
    # inv_th_im = cv2.cvtColor(inv_th_im, cv2.COLOR_BGR2RGB)
    inv_th_im[pro_im < 255] = 0
    # Contour Detection

    crop_img = inv_th_im.copy()

    cont_im = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)).copy()
    # cont_im  = cont_im[:,:,0]
    cont_im = cv2.GaussianBlur(cont_im, (Filter_sz, Filter_sz), 20)

    # Find Laplacian/Canny edges
    if edge_filter == 'Laplacian':
        edged = cv2.Laplacian(cont_im, cv2.CV_8UC1, ksize=5)
    elif edge_filter == 'Canny':
        edged = cv2.Canny(cont_im, 20, 100, 5)  # 30 70
    elif edge_filter == 'Old_Adaptive':

        w = adaptive_filter_size
        k = 0.002  # 0.2-0.5

        kernel = np.ones((w, w), np.float32) / (w * w)
        mean = cv2.filter2D(cont_im, -1, kernel)
        mean_sqr = cv2.filter2D(np.square(cont_im), -1, kernel)
        std = mean_sqr - np.square(mean)
        one = np.ones(cont_im.shape)
        R = np.max(std)
        threshold = mean * (one + k * (std / R - one))
        edged = np.uint8(255 * (cont_im > threshold))
    elif edge_filter == 'Adaptive STD':
        crop_img = inv_th_im.copy()

        cont_im = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)).copy()
        cont_im = np.float64(cont_im[:, :, 0])
        cont_im = cv2.GaussianBlur(cont_im, (Filter_sz, Filter_sz), 20)

        w = adaptive_filter_size
        k = 1  # 0.2-0.5 # 0.002

        kernel = np.ones((w, w), np.float32) / (w * w)
        mean = cv2.filter2D(cont_im, -1, kernel)
        mean_sqr = cv2.filter2D(np.square(cont_im), -1, kernel)
        std = np.sqrt(np.clip((mean_sqr - np.square(mean)), 0, 9999))

        std = np.uint8(255 * std / np.max(std))
        edged = std

    elif edge_filter == 'Adaptive Thresholding':
        crop_img = inv_th_im.copy()

        cont_im = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)).copy()
        cont_im = np.float64(cont_im[:, :, 0])
        cont_im = cv2.GaussianBlur(cont_im, (Filter_sz, Filter_sz), 20)

        w = adaptive_filter_size
        k = 1  # 0.2-0.5 # 0.002

        kernel = np.ones((w, w), np.float32) / (w * w)
        mean = cv2.filter2D(cont_im, -1, kernel)
        mean_sqr = cv2.filter2D(np.square(cont_im), -1, kernel)
        std = np.sqrt(np.clip((mean_sqr - np.square(mean)), 0, 9999))

        one = np.ones(cont_im.shape)
        R = np.max(std)
        threshold = mean * (one + k * (std / R - one))

        edged = np.uint8(255*(cont_im > threshold))
    elif edge_filter == 'Adaptive Gradient Thresholding':
        crop_img = inv_th_im.copy()

        cont_im = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)).copy()
        cont_im = np.float64(cont_im[:, :, 0])
        cont_im = cv2.GaussianBlur(cont_im, (Filter_sz, Filter_sz), 20)

        w = adaptive_filter_size
        k = 1  # 0.2-0.5 # 0.002

        kernel = np.ones((w, w), np.float32) / (w * w)
        mean = cv2.filter2D(cont_im, -1, kernel)
        mean_sqr = cv2.filter2D(np.square(cont_im), -1, kernel)
        std = np.sqrt(np.clip((mean_sqr - np.square(mean)), 0, 9999))

        one = np.ones(cont_im.shape)
        R = np.max(std)
        threshold = mean * (one + k * (std / R - one))
        sobelx = cv2.Sobel(cont_im, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(cont_im, cv2.CV_64F, 0, 1, ksize=5)
        mag_sobel = np.sqrt(sobelx * sobelx + sobely * sobely)
        edged = np.uint8(255*(mag_sobel > threshold))

    else:
        print("Invalid Edge Filter Type")
        return None, None, None
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = list(contours)  # convert the contours tuple into a list.

    # Contour Filtering

    biggest_cont_img = inv_th_im.copy()
    biggest_cont_colored_img = im_color.copy()
    biggest_cont_colored_img = (cv2.cvtColor(
        biggest_cont_colored_img, cv2.COLOR_BGR2HSV))
    contours.sort(key=cv2.contourArea, reverse=True)

    L_tolerans = 0.9

    max_len = 0
    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        if length > max_len:
            biggest = cnt
            max_len = length

    big_conts = []

    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        if (max_len - length) / max_len < L_tolerans:
            big_conts.append(cnt)
    print("Number of Leafs Found {0}".format(len(big_conts)))
    if contour_font == None:
        contour_font = int(0.01*np.min(im.shape))

    cv2.drawContours(biggest_cont_img, big_conts, -
                     1, (0, 255, 0), contour_font)

    for i in range(len(big_conts)):
        L = len(big_conts)

        cv2.drawContours(biggest_cont_colored_img, [
                         big_conts[i]], -1, ((180) * i / L, 255, 255), contour_font)
    biggest_cont_colored = (cv2.cvtColor(
        biggest_cont_colored_img, cv2.COLOR_HSV2BGR))

    return big_conts, biggest_cont_colored, inv_th_im


def Label_img(img_name, infilename, outfilename):
    img_adress = infilename + img_name
    im_color = cv2.imread(img_adress)

    print(img_adress)
    tot_width = im_color.shape[0]
    tot_height = im_color.shape[1]

    conts, _, seg_img = PreProcc(im_color)

    file1 = open(outfilename + img_name[:(len(img_adress) - 4)] + ".txt", "w")

    for cnt in conts:
        xmin = np.min(cnt[:, 0, 0])
        xmax = np.max(cnt[:, 0, 0])
        ymin = np.min(cnt[:, 0, 1])
        ymax = np.max(cnt[:, 0, 1])
        xcenter = (xmax + xmin) / (2 * tot_width)
        ycenter = (ymax + ymin) / (2 * tot_height)
        width = (xmax - xmin) / tot_width
        height = (ymax - ymin) / tot_height

        cv2.rectangle(im_color, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        L = [str(0) + " ", str(xcenter) + " ", str(ycenter) +
             " ", str(width) + " ", str(height) + "\n"]
        file1.writelines(L)

    cv2.imwrite(outfilename + img_name, seg_img)
    file1.close()
    return im_color


def Leaf_Segment(img, th=0.7):
    im_color_16 = img.copy().astype('float64')
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CIVE thresholding
    cive_im = 0.441 * im_color_16[:, :, 2] - 0.811 * im_color_16[:, :, 1] + 0.385 * im_color_16[:, :,
                                                                                                0] + 18.78745 * np.ones(im.shape)
    mcive_im = (cive_im - np.ones(cive_im.shape) * np.min(cive_im)
                ) / (np.max(cive_im) - np.min(cive_im))

    new_im = np.zeros(im.shape)
    indx = (mcive_im < th)
    new_im[indx] = 255

    # Dilation + Erosion
    kernel_type = "Gaussian"
    if kernel_type == "Gaussian":
        kernel = np.outer(cv2.getGaussianKernel(5, 1),
                          cv2.getGaussianKernel(5, 1))
    else:
        kernel = np.ones((9, 9), np.uint8)

    img_dilation = cv2.dilate(new_im, kernel, iterations=3)
    img_diler = cv2.erode(img_dilation, kernel, iterations=3)

    pro_im = new_im
    for i in range(5):
        pro_im = cv2.erode(pro_im, kernel, iterations=1)
        pro_im = cv2.dilate(pro_im, kernel, iterations=1)

    # Inverse Thresholding
    inv_th_im = img.copy()  # cv2.imread("drive/My Drive/EE493/Ripeness/input.jpg")
    inv_th_im = cv2.cvtColor(inv_th_im, cv2.COLOR_BGR2RGB)
    inv_th_im[pro_im < 255] = 0

    return inv_th_im, indx


'''
Ripeness state =
0:ripe
1:semi-ripe
2:unripe
3:none
'''


def detect_ripeness(img, therholds=None):
    _, cont_img, seg_img = PreProcc(
        img, adaptive_filter_size=5, th=0.53, edge_filter='Adaptive Gradient Thresholding')
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    im_area = im.shape[0] * im.shape[1]

    binary_im = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    binary_im = np.ones(binary_im.shape) * (binary_im > 0)
    total_area = 100 * np.sum(binary_im) / im_area
    p_img = cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB)
    # cv2.drawContours(p_img,conts,-1,(255,0,0),15)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if therholds == None:
        therholds = [35, 15, 0.1]
    if total_area > therholds[0]:
        ripeness_state = 0
        cv2.putText(p_img, 'Ripe', (10, 450), font,
                    15, (0, 0, 255), 45, cv2.LINE_AA)

    elif total_area > therholds[1]:  # 25
        ripeness_state = 1
        cv2.putText(p_img, 'Semi-Ripe', (10, 450), font,
                    15, (0, 0, 255), 45, cv2.LINE_AA)

    elif total_area > therholds[2]:  # 5
        ripeness_state = 2
        cv2.putText(p_img, 'Unripe', (10, 450), font,
                    15, (0, 0, 255), 45, cv2.LINE_AA)

    else:
        ripeness_state = 3
        cv2.putText(p_img, 'None', (10, 450), font,
                    15, (0, 0, 255), 45, cv2.LINE_AA)

    print("Total Area found is {0}".format(total_area))

    return ripeness_state, total_area, p_img

##########DISEASE##########

# TEXTURE
# GlCM
# indicies:
# 0:corrolation, 1: dissimilarity, 2: contrast, 3: homogeneity, 4: energy


def GLCM(img, dist=5, angle=0):
    feats = []
    glcm = greycomatrix(img, distances=[dist], angles=[angle], levels=256,
                        symmetric=True, normed=True)
    feats.append(greycoprops(glcm, 'correlation')[0, 0])
    feats.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    feats.append(greycoprops(glcm, 'contrast')[0, 0])
    feats.append(greycoprops(glcm, 'homogeneity')[0, 0])
    feats.append(greycoprops(glcm, 'energy')[0, 0])

    return feats


# Gabor Features 0:mean, 1:std
# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

def Gabor(img, ksize=21, sigma=8.0, theta=np.pi / 4, lamda=10.0, gamma=0.5, psi=0):
    g_kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)

    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    feats = []
    feats.append(np.mean(filtered_img))
    feats.append(np.std(filtered_img))

    return feats


# COLOR

# RGB Mean&STD
# Indicies 0-2:rgb mean, 3-5: rgb std
def rgb_mean_std(img_color, color_space='BGR'):
    if color_space == 'BGR':
        rm = np.mean(img_color[:, :, 2])
        gm = np.mean(img_color[:, :, 1])
        bm = np.mean(img_color[:, :, 0])

        rs = np.std(img_color[:, :, 2])
        gs = np.std(img_color[:, :, 1])
        bs = np.std(img_color[:, :, 0])

    elif color_space == 'RGB':
        rm = np.mean(img_color[:, :, 0])
        gm = np.mean(img_color[:, :, 1])
        bm = np.mean(img_color[:, :, 2])

        rs = np.std(img_color[:, :, 0])
        gs = np.std(img_color[:, :, 1])
        bs = np.std(img_color[:, :, 2])

    feats = [rm, gm, bm, rs, gs, bs]
    return feats


def YgCbCr_features(img_color, basic=True):
    img_ygcbr = cv2.cvtColor(img_color, cv2.COLOR_BGR2YCrCb)

    feats = []
    # mean
    for i in range(3):
        feats.append(np.mean(img_ygcbr[:, :, i]))
    # std
    for i in range(3):
        feats.append(np.std(img_ygcbr[:, :, i]))
    if basic == False:
        # Median
        for i in range(3):
            feats.append(np.median(img_ygcbr[:, :, i]))
        # Quarter1
        for i in range(3):
            feats.append(np.quantile(img_ygcbr[:, :, i], 0.25))
            # Quarter3
        for i in range(3):
            feats.append(np.quantile(img_ygcbr[:, :, i], 0.75))
    return feats


def AvgBrigthnessInfo(img_color, color_space='BGR'):
    if color_space == 'BGR':
        rm = np.mean(img_color[:, :, 2])
        gm = np.mean(img_color[:, :, 1])
        bm = np.mean(img_color[:, :, 0])

    elif color_space == 'RGB':
        rm = np.mean(img_color[:, :, 0])
        gm = np.mean(img_color[:, :, 1])
        bm = np.mean(img_color[:, :, 2])

    abi = 0.299 * rm + 0.587 * gm + 0.114 * bm
    return abi


# Get Features GLCM,Gabor,RGB Mean-STD,Average Brightness Information
# Indicies
'''
0:corrolation (gray)
1: dissimilarity (gray)
2: contrast (gray)
3: homogeneity (gray)
4: energy (gray)
5: Gabor Mean 
6: Gabor STD
7-9: YgCbCr Mean
10-12: YgCbCr STD
13-15: YgCbCr Median
16-18: YgCbCr Quartile 1
19-21: YgCbCr Quartile 1
22-27: Hue CCM
28-32: Saturation CCM
33-37: Intensity CCM
38: Average Brightness Information
'''


def get_all_feats(img_color, yg=True, basic=True, extended=True):
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    feats = GLCM(img)
    a = Gabor(img)
    feats = np.concatenate((feats, a), axis=0)

    if yg:
        a = YgCbCr_features(img_color, basic)
    else:
        a = rgb_mean_std(img_color)
    feats = np.concatenate((feats, a), axis=0)
    if extended:
        img_hsi = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        a = GLCM(img_hsi[:, :, 0])
        feats = np.concatenate((feats, a), axis=0)
        a = GLCM(img_hsi[:, :, 1])
        feats = np.concatenate((feats, a), axis=0)
        a = GLCM(img_hsi[:, :, 2])
        feats = np.concatenate((feats, a), axis=0)

    a = AvgBrigthnessInfo(img_color)
    feats = np.concatenate((feats, [a]), axis=0)
    return feats


# Gets the features in get_all_feats for a batch of images
def get_feats_patch(img_patch, yg=True, basic=True, extended=True):
    feat_patch = []
    for img in img_patch:
        feat_patch.append(get_all_feats(
            img, yg=yg, basic=basic, extended=extended))

    return feat_patch


def create_feat_names(color='Y', basic=True, extended=True):
    names = []
    GLCM_names = ["Corrolation", "Dissimilarity",
                  "Contrast", "Homogeneity", "Energy"]
    m_s = ['Mean', 'STD']
    for i in range(5):
        names.append('Gray ' + GLCM_names[i])

    names.append("Gabor Mean")
    names.append("Gabor STD")

    if color == 'Y':
        colors = ['Yg ', 'Cb ', 'Cr ']
    else:
        colors = ['red ', 'green ', 'blue ']

    for i in range(2):
        for j in range(3):
            names.append(colors[j] + m_s[i])

    if (color == 'Y') & (basic == False):
        med_q = ['Median', 'Quartile 1', 'Quartile 3']
        for i in range(3):
            for j in range(3):
                names.append(colors[j] + med_q[i])

    if extended:
        hsv_names = ['Hue ', 'Saturation ', 'Intensity ']
        for i in range(3):
            for j in range(5):
                names.append(hsv_names[i] + GLCM_names[j])

    names.append('ABI')
    return names


##Implemetation##

def read_csv(file_name):
    file = open(file_name)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    sampleArray = np.array(rows)
    convertedArray = sampleArray.astype(np.float)
    return convertedArray


def load_model(dir):
    knn = pickle.load(open(dir+'/knnpickle_file', 'rb'))
    return knn


'''
y_pred =
0:Bacterial Blight
1:Mildew Downy
2:Rust
3:Gray Spot
4:Brown Spot
5:Healthy
'''


def eval_disease(img, dir):

    conts, cont_img, seg_img = PreProcc(img, adaptive_filter_size=5, th=0.53,
                                        edge_filter='Adaptive Gradient Thresholding')

    # Hull

    hull = []

    for cnt in conts:
        new_hull = cv2.convexHull(cnt)
        hull.append(new_hull)

    main_mask = np.zeros(seg_img.shape, dtype=np.uint8)
    for pts in hull:
        # print(pts[:,0])
        mask = np.zeros(seg_img.shape, dtype=np.uint8)
        # roi_corners = np.array(pts, dtype=np.int32) #pointsOf the polygon Like [[(10,10), (300,300), (10,300)]]
        cv2.fillPoly(mask, [pts[:, 0]], (255, 255, 255))

        main_mask = cv2.bitwise_or(main_mask, mask)
        # apply the mask

    masked_image = cv2.bitwise_and(img, main_mask)

    test_feat = get_all_feats(masked_image, extended=False)
    knn = load_model(dir)
    y_pred = knn.predict([test_feat])
    return y_pred

# Testing Disease


def load_features(dir):
    class_names = ['Bacterial Blight', 'Mildew Downy',
                   'Rust', 'Gray Spot', 'Brown Spot', 'Healthy']
    num_of_classes = len(class_names)

    features = read_csv(dir+'/features.csv')
    class_features = []

    for j in range(num_of_classes):
        class_features.append(read_csv(dir+'/class' + str(j) + '.csv'))

    knn = pickle.load(open(dir+'/knnpickle_file', 'rb'))
    return knn, features, class_features


def test_knn(dir, size=0.2, show_wrongs=False, img_patches=None):
    knn, features, class_features = load_features(dir)
    tot_len = features.shape[0]
    class_names = ['Bacterial Blight', 'Mildew Downy',
                   'Rust', 'Gray Spot', 'Brown Spot', 'Healthy']
    num_of_classes = len(class_names)
    #names = create_feat_names(basic=False)
    Lengths = []
    for i in range(num_of_classes):
        Lengths.append(class_features[i].shape[0])

    prev_sum = 0
    Label = np.zeros((tot_len, 1))
    for i in range(num_of_classes):
        Label[prev_sum:prev_sum + Lengths[i]] = i
        prev_sum += Lengths[i]

    #size = 0.6
    val_num = []
    train_num = []
    new_indexes = []
    Label_train = []
    Label_test = []
    features_train = []
    features_test = []
    train_set = []
    val_set = []
    for i in range(num_of_classes):
        val_num.append(int(np.ceil(Lengths[i] * size)))
        train_num.append(Lengths[i] - val_num[i])
        new_indx = np.random.permutation(Lengths[i])
        new_indexes.append(new_indx)
        train_set.append(class_features[i][new_indx[val_num[i]:], :])
        val_set.append(class_features[i][new_indx[:val_num[i]], :])
        arr = [i for j in range(train_num[i])]
        Label_train = np.concatenate((Label_train, arr), axis=0)
        arr = [i for j in range(val_num[i])]
        Label_test = np.concatenate((Label_test, arr), axis=0)

        if features_train == []:
            features_train = train_set[i]
            features_test = val_set[i]
        else:
            features_train = np.concatenate(
                (features_train, train_set[i]), axis=0)
            features_test = np.concatenate((features_test, val_set[i]), axis=0)

    y_pred = knn.predict(features_test)
    err = Label_test - y_pred

    accuracy = []
    prev_sum = 0
    for i in range(num_of_classes):
        accuracy.append(
            np.sum(y_pred[prev_sum:prev_sum + val_num[i]] == i) / val_num[i])
        print("{0} accuracy: {1}".format(class_names[i], accuracy[i]))
        prev_sum += val_num[i]
    tot_accuracy = np.sum(err == 0) / np.sum(val_num)
    print("Total Accuracy: {0}".format(tot_accuracy))

    conf_mtx = confusion_matrix(Label_test, y_pred)
    print(conf_mtx)
    if show_wrongs:
        for i in range(num_of_classes):

            for j in range(val_num[i]):
                if y_pred[prev_sum + j] != i:
                    print("{0} no:{1}, confused with:{2}".format(class_names[i], new_indexes[i][j],
                                                                 class_names[int(y_pred[prev_sum + j])]))
                    img = cv2.cvtColor(
                        img_patches[i][new_indexes[i][j]], cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.show()
            prev_sum += val_num[i]
