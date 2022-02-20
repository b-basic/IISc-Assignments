# # input image size > 100x100 takes a lot of time to run, preferably keep size of image small

import skimage.io
from matplotlib import pyplot as plt
from colormap import rgb2hex
import numpy as np
from math import exp
from scipy.sparse.linalg import eigsh


# set the following params within the function:
# sigma_i, sigma_j, cut_off distance
def get_similarity_brightness_distance(pixel1, index1, pixel2, index2, w):
    sigma_i = 10
    sigma_x = 15
    cut_off_dist = 200

    pixel11_i = int(index1/w)
    pixel1_j = int(index1//w)
    pixel2_i = int(index2/w)
    pixel2_j = int(index2//w)

    eucd_dist = ((pixel11_i - pixel2_i)**2 + (pixel1_j - pixel2_j)**2)**0.5

    pixel1_bright = np.sum(pixel1)/3
    pixel2_bright = np.sum(pixel2)/3

    bright_dist = abs(pixel1_bright - pixel2_bright)

    if eucd_dist < cut_off_dist:
        wij = exp(-bright_dist/sigma_i) * exp(-(eucd_dist**2)/sigma_x)
        return wij

    return 0


def get_similarity_color(pixel1, pixel2):
    # getting hex value of pixel color
    p1_hex = "0x" + rgb2hex(pixel1[0], pixel1[1], pixel1[2])[1:]
    p2_hex = "0x" + rgb2hex(pixel2[0], pixel2[1], pixel2[2])[1:]

    # converting hex to decimal
    p1_value = int(p1_hex, 16)
    p2_value = int(p2_hex, 16)
    max_p_value = int("0xFFFFFF", 16)

    # smaller hex value distance = more similarity
    distance = max_p_value - abs(p1_value - p2_value)

    return distance


# set similarity measure inside function by choosing appropriate function (by commenting/uncommenting)
def get_W(image):
    print("in")

    R = image[:, :, 0].flatten()
    G = image[:, :, 1].flatten()
    B = image[:, :, 2].flatten()

    pixels_flattened = np.zeros([R.shape[0], 3]).astype(int)

    pixels_flattened[:, 0] = R
    pixels_flattened[:, 1] = G
    pixels_flattened[:, 2] = B

    W = np.zeros([R.shape[0], R.shape[0]])

    # set similarity measure by commenting below
    for i in range(R.shape[0]):
        for j in range(i+1):
            # W[i, j] = get_similarity_color(pixels_flattened[i], pixels_flattened[j])
            W[i, j] = get_similarity_brightness_distance(pixels_flattened[i], i, pixels_flattened[j], j, image.shape[1])
            W[j, i] = W[i, j]

    print("out")
    return W


# input_img = 'Ncut_test.png'
input_img = 'natural_im1_small.jpg'
# input_img = 'Ncut_test3.png'

img = (skimage.io.imread(input_img)[:, :, :3] * 255).astype(int)
# plt.imshow(img)
# plt.show()
# print(img.shape)

W = get_W(img)
# np.save('W.npy', W)     # save
# W = np.load('W.npy')    # load

D = np.zeros(W.shape)
for i in range(W.shape[0]):
    D[i, i] = np.sum(W[i, :])

A = D - W
B = D
eigvals, eigvecs = eigsh(A, 2, M=B, sigma=None, which='SM')
second_smallest = eigvecs[:, 1] > 0

segmented_img = second_smallest.reshape((img.shape[0], img.shape[1]))
plt.imshow(segmented_img, cmap='gray')
plt.show()
