# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def proc01():
    # 读取图像
    img = cv.imread('../data/lena.png', 0)

    # 快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)

    # 默认结果中心点位置是在左上角,
    # 调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    # fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))

    # 展示结果
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
    plt.axis('off')
    plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
    plt.axis('off')
    plt.show()


def proc02():
    # 读取图像
    img = cv.imread('../data/lena.png', 0)

    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    res = np.log(np.abs(fshift))

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    # 展示结果
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
    plt.axis('off')
    plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
    plt.axis('off')
    plt.show()


def proc03():
    # 读取图像
    img = cv.imread('../data/lena.png', 0)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

    # 将频谱低频从左上角移动至中心位置
    dft_shift = np.fft.fftshift(dft)

    # 频谱图像双通道复数转换为0-255区间
    result = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # 显示图像
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def proc04():
    # 读取图像
    img = cv.imread('../data/lena.png', 0)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dftshift = np.fft.fftshift(dft)
    res1 = 20 * np.log(cv.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(dftshift)
    iimg = cv.idft(ishift)
    res2 = cv.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    # 显示图像
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')
    plt.axis('off')
    plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')
    plt.axis('off')
    plt.show()


def proc05():
    # 读取图像
    img = cv.imread('../data/lena.png', 0)

    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 设置高通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    # 显示原始图像和高通滤波处理图像
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title('Result Image')
    plt.axis('off')
    plt.show()


def proc06():
    # 读取图像
    img = cv.imread('../data/lena.png', 0)

    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask
    print(f.shape, fshift.shape, mask.shape)

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv.idft(ishift)
    res = cv.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    # 显示原始图像和低通滤波处理图像
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Result Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    proc06()
