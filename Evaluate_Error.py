import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
from sklearn.metrics import mutual_info_score


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def evaluat_error(sp, act):
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros(len(x))
    abs_r = np.zeros(len(x))
    abs_x = np.zeros(len(x))
    abs_r_x = np.zeros(len(x))
    abs_x_r = np.zeros(len(x))
    abs_r_x__r = np.zeros(len(x))
    for j in range(1, len(x)):
        points[j] = abs(x[j] - x[j - 1])
    for i in range(len(r)):
        abs_r[i] = abs(r[i])
    for i in range(len(r)):
        abs_x[i] = abs(x[i])
    for i in range(len(r)):
        abs_r_x[i] = abs(r[i] - x[i])
    for i in range(len(r)):
        abs_x_r[i] = abs(x[i] - r[i])
    for i in range(len(r)):
        abs_r_x__r[i] = abs((r[i] - x[i]) / r[i])
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    MSE = np.square(np.subtract(sp, act)).mean()

    SSIM = ssim(sp, act, multichannel=True)  # Structural Similarity Index

    PSNR = psnr(sp, act)  # Peak Signal to Noise Ratio

    Mutual_Information = mutual_info_score(sp, act)

    EVAL_ERR = [PSNR, SSIM, MSE, Mutual_Information, rmse]
    return EVAL_ERR
