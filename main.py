import cv2
import numpy as np
from skimage import color
from scipy import ndimage
from scipy.signal import convolve2d

# Importing Image
org_img = cv2.imread('drum.jpg')
img = org_img.copy()
cv2.imshow('Input image', org_img)

# Red compensation
def redCompensate(img, window):
    alpha = 1
    r = img[:,:,2].astype(float) / 255
    g = img[:,:,1].astype(float) / 255
    b = img[:,:,0].astype(float) / 255

    height, width, _ = img.shape
    padsize = ((window-1)//2, (window-1)//2)
    padr = np.pad(r, padsize, mode='symmetric')
    padg = np.pad(g, padsize, mode='symmetric')

    ret = img.copy()
    for i in range(height):
        for j in range(width):
            slider = padr[i:i+window, j:j+window]
            slideg = padg[i:i+window, j:j+window]
            r_mean = np.mean(slider)
            g_mean = np.mean(slideg)
            Irc = r[i,j] + alpha * (g_mean - r_mean) * (1-r[i,j]) * g[i,j]
            Irc = int(Irc * 255)
            ret[i, j, 2] = Irc
    return ret

red_comp_img = redCompensate(img, 5)
cv2.imshow('Red Compensated', red_comp_img)

# White balancing
def gray_balance(image):
    L = 255
    r, g, b = cv2.split(image)
    Ravg, Gavg, Bavg = np.mean(r), np.mean(g), np.mean(b)

    Max = max(Ravg, Gavg, Bavg)
    ratio = [Max / Ravg, Max / Gavg, Max / Bavg]

    satLevel = 0.005 * np.array(ratio)

    m, n, p = image.shape
    imgRGB_orig = np.array([r.flatten(), g.flatten(), b.flatten()])

    imRGB = np.zeros_like(imgRGB_orig)

    for ch in range(p):
        q = [satLevel[ch], 1 - satLevel[ch]]
        tiles = np.quantile(imgRGB_orig[ch], q)
        temp = np.clip(imgRGB_orig[ch], tiles[0], tiles[1])
        pmin, pmax = temp.min(), temp.max()
        imRGB[ch] = (temp - pmin) * L / (pmax - pmin)

    output = np.zeros_like(image)
    for i in range(p):
        output[:, :, i] = imRGB[i].reshape((m, n))

    return output.astype(np.uint8)

wb_img = gray_balance(red_comp_img)
cv2.imshow('White Balanced Image', wb_img)

# Gamma correction
def gammaCorrection(img, alpha, gamma):
    return (alpha * (img.astype(float) / 255) ** gamma * 255).astype(np.uint8)

alpha = 1
gamma = 1.2
gamma_crct_img = gammaCorrection(wb_img, alpha, gamma)
cv2.imshow('Gamma corrected White balance image', gamma_crct_img)

# Sharpen
def sharp(img):
    img = img.astype(float) / 255
    GaussKernel = cv2.getGaussianKernel(5, 3)
    GaussKernel = GaussKernel @ GaussKernel.T
    imBlur = cv2.filter2D(img, -1, GaussKernel)
    unSharpMask = img - imBlur
    stretchIm = hisStretching(unSharpMask)
    result = (img + stretchIm) / 2
    return (result * 255).astype(np.uint8)

def hisStretching(img):
    img = img.astype(float)
    for i in range(3):
        channel = img[:,:,i]
        minVal, maxVal = channel.min(), channel.max()
        img[:,:,i] = (channel - minVal) / (maxVal - minVal)
    return img

sharpen_img = sharp(wb_img)
cv2.imshow('Sharpen White balance image', sharpen_img)

# Calculating weights
def rgb_to_lab(rgb):
    return color.rgb2lab(rgb[:,:,::-1])

gamma_img_lab = rgb_to_lab(gamma_crct_img)
gamma_img_lab_1 = gamma_img_lab[:,:,0] / 255
sharpen_img_lab = rgb_to_lab(sharpen_img)
sharpen_img_lab1 = sharpen_img_lab[:,:,0] / 255

# 1 Laplacian contrast weight
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
WL1 = np.abs(convolve2d(gamma_img_lab_1, laplacian, mode='same', boundary='symm'))
WL2 = np.abs(convolve2d(sharpen_img_lab1, laplacian, mode='same', boundary='symm'))

# 2 Saliency weight
def saliency_detection(img):
    gfrgb = cv2.GaussianBlur(img, (3, 3), 3)
    lab = color.rgb2lab(gfrgb[:,:,::-1])
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    lm, am, bm = l.mean(), a.mean(), b.mean()
    return (l-lm)**2 + (a-am)**2 + (b-bm)**2

WS1 = saliency_detection(gamma_crct_img)
WS2 = saliency_detection(sharpen_img)

# 3 Saturation weight
def Saturation_weight(img):
    lab = rgb_to_lab(img) / 255
    return np.sqrt(1/3 * ((img[:,:,0] - lab[:,:,0])**2 +
                          (img[:,:,1] - lab[:,:,0])**2 +
                          (img[:,:,2] - lab[:,:,0])**2))

WSat1 = Saturation_weight(gamma_crct_img)
WSat2 = Saturation_weight(sharpen_img)

# Normalized weight
def norm_weight(w1, w2, w3, w4, w5, w6):
    K = 2
    delta = 0.1
    nw1 = w1 + w2 + w3
    nw2 = w4 + w5 + w6
    w = nw1 + nw2
    nw1 = (nw1 + delta) / (w + K * delta)
    nw2 = (nw2 + delta) / (w + K * delta)
    return nw1, nw2

W1, W2 = norm_weight(WL1, WS1, WSat1, WL2, WS2, WSat2)

# Image Fusion
level = 3

def gaussian_pyramid(img, level):
    h = np.array([1, 4, 6, 4, 1]) / 16
    filt = h[:, np.newaxis] @ h[np.newaxis, :]
    out = [cv2.filter2D(img, -1, filt)]
    temp_img = img
    for _ in range(1, level):
        temp_img = temp_img[::2, ::2]
        out.append(cv2.filter2D(temp_img, -1, filt))
    return out

def laplacian_pyramid(img, level):
    h = np.array([1, 4, 6, 4, 1]) / 16
    out = [img]
    temp_img = img
    for _ in range(1, level):
        temp_img = temp_img[::2, ::2]
        out.append(temp_img)
    for i in range(level - 1):
        m, n = out[i].shape
        out[i] = out[i] - cv2.resize(out[i+1], (n, m))
    return out

# weight gaussian pyramid
Weight1 = gaussian_pyramid(W1, level)
Weight2 = gaussian_pyramid(W2, level)

# image laplacian pyramid
r1 = laplacian_pyramid(gamma_crct_img[:, :, 2].astype(float), level)
g1 = laplacian_pyramid(gamma_crct_img[:, :, 1].astype(float), level)
b1 = laplacian_pyramid(gamma_crct_img[:, :, 0].astype(float), level)

r2 = laplacian_pyramid(sharpen_img[:, :, 2].astype(float), level)
g2 = laplacian_pyramid(sharpen_img[:, :, 1].astype(float), level)
b2 = laplacian_pyramid(sharpen_img[:, :, 0].astype(float), level)

# Fusion
R_r = [Weight1[i] * r1[i] + Weight2[i] * r2[i] for i in range(level)]
G_g = [Weight1[i] * g1[i] + Weight2[i] * g2[i] for i in range(level)]
B_b = [Weight1[i] * b1[i] + Weight2[i] * b2[i] for i in range(level)]

def pyramid_reconstruct(pyramid):
    for i in range(len(pyramid)-1, 0, -1):
        m, n = pyramid[i-1].shape
        pyramid[i-1] += cv2.resize(pyramid[i], (n, m))
    return pyramid[0]

R = pyramid_reconstruct(R_r)
G = pyramid_reconstruct(G_g)
B = pyramid_reconstruct(B_b)

fusion = cv2.merge([B, G, R])

final_result = fusion.astype(np.uint8)
cv2.imshow("Result [Fusion image]", final_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

