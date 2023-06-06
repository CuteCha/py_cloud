# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def test1():
    '''
    离散傅里叶变换
    一维时序信号y，它由2V的直流分量(0Hz)，和振幅为3V，频率为50Hz的交流信号，以及振幅为1.5V，频率为75Hz的交流信号组成：
    y = 2 + 3*np.cos(2*np.pi*50*t) + 1.5*np.cos(2*np.pi*75*t)
    然后我们采用256Hz的采样频率，总共采样256个点。
    '''
    fs = 256  # 采样频率， 要大于信号频率的两倍
    t = np.arange(0, 1, 1.0 / fs)  # 1秒采样fs个点
    N = len(t)
    freq = np.arange(N)  # 频率counter

    # x = 2 + 3 * cos(2 * pi * 50 * t) + 1.5 * cos(2 * pi * 75 * t)  # 离散化后的x[n]
    x = 2 + 3 * np.cos(2 * np.pi * 10 * t) + 1.5 * np.cos(2 * np.pi * 15 * t)  # 离散化后的x[n]

    X = np.fft.fft(x)  # 离散傅里叶变换

    '''
    根据STFT公式原理，实现的STFT计算，做了/N的标准化
    '''
    X2 = np.zeros(N, dtype=np.complex)  # X[n]
    for k in range(0, N):  # 0,1,2,...,N-1
        for n in range(0, N):  # 0,1,2,...,N-1
            # X[k] = X[k] + x[n] * np.exp(-2j * pi * k * n / N)
            X2[k] = X2[k] + (1 / N) * x[n] * np.exp(-2j * np.pi * k * n / N)

    fig, ax = plt.subplots(5, 1, figsize=(12, 12))

    # 绘制原始时域图像
    ax[0].plot(t, x, label='原始时域信号')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(freq, abs(X), 'r', label='调用np.fft库计算结果')
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()

    ax[2].plot(freq, abs(X2), 'r', label='根据STFT计算结果')
    ax[2].set_xlabel('Freq (Hz)')
    ax[2].set_ylabel('Amplitude')
    ax[2].legend()

    X_norm = X / (N / 2)  # 换算成实际的振幅
    X_norm[0] = X_norm[0] / 2
    ax[3].plot(freq, abs(X_norm), 'r', label='转换为原始信号振幅')
    ax[3].set_xlabel('Freq (Hz)')
    ax[3].set_ylabel('Amplitude')
    ax[3].set_yticks(np.arange(0, 3))
    ax[3].legend()

    freq_half = freq[range(int(N / 2))]  # 前一半频率
    X_half = X_norm[range(int(N / 2))]

    ax[4].plot(freq_half, abs(X_half), 'b', label='前N/2个频率')
    ax[4].set_xlabel('Freq (Hz)')
    ax[4].set_ylabel('Amplitude')
    ax[4].set_yticks(np.arange(0, 3))
    ax[4].legend()

    plt.show()


def test2():
    sampling_rate = 8096  # 采样率
    fft_size = 1024  # FFT长度
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    x = np.sin(2 * np.pi * 156.25 * t) + 2 * np.sin(2 * np.pi * 234.375 * t) + 3 * np.sin(2 * np.pi * 200 * t)
    xs = x[:fft_size]

    xf = np.fft.rfft(xs) / fft_size  # 返回fft_size/2+1 个频率

    freqs = np.linspace(0, sampling_rate // 2, fft_size // 2 + 1)  # 表示频率
    xfp = np.abs(xf) * 2  # 代表信号的幅值，即振幅

    plt.figure(num='original', figsize=(15, 6))
    plt.plot(x[:fft_size])

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t[:fft_size], xs)
    plt.xlabel("t(s)")
    plt.title("156.25Hz and 234.375Hz waveform and spectrum")

    plt.subplot(212)
    plt.plot(freqs, xfp)
    plt.xlabel("freq(Hz)")
    plt.ylabel("amplitude")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def test3():
    N = 1024
    t = np.linspace(0, 2 * np.pi, N)
    x = 0.3 * np.cos(t) + 0.5 * np.cos(2 * t + np.pi / 4) + 0.8 * np.cos(3 * t - np.pi / 3)
    xf = np.fft.fft(x) / N
    freq = np.arange(N)

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, x)
    plt.xlabel("t(s)")
    plt.title("waveform and spectrum")

    plt.subplot(212)
    plt.plot(freq, np.abs(xf) * 2)
    plt.xlabel("freq(Hz)")
    plt.ylabel("amplitude")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def test4():
    Fs = 64
    f_o = [5, 15, 20]
    t = np.arange(0, 10, 1.0 / Fs)
    x = np.sum([np.sin(2 * np.pi * f1 * t) for f1 in f_o], axis=0)
    N = len(t)

    X = np.fft.fft(x)
    f = np.arange(N) * Fs * 1.0 / N

    f_shift = f - Fs / 2
    X_shift = np.fft.fftshift(X)  # 调整0频位置

    N_p = N // 2
    f_p = f_shift[N_p:]
    X_p = (np.abs(X_shift)[N_p:]) * 2

    x_r = np.fft.ifft(X)

    plt.figure(figsize=(8, 20))
    plt.subplot(511)
    plt.plot(t, x)
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Signal")

    plt.subplot(512)
    plt.plot(f, np.abs(X))
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("spectrum")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(513)
    plt.plot(f_shift, np.abs(X_shift))
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("spectrum")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(514)
    plt.plot(f_p, X_p)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("spectrum")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(515)
    plt.plot(t, np.real(x_r))
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("reconstruct Signal")

    plt.show()


def test5():
    F = 30
    Fs = 256
    Ts = 1 / Fs

    tc = np.arange(0, 5 / F, 1e-4)
    xc = np.cos(2 * np.pi * F * tc)
    td = np.arange(0, 5 / F, Ts)
    xd = np.cos(2 * np.pi * F * td)
    N = len(td)

    xr = np.sum([xd[i] * np.sinc(tc / Ts - i) for i in range(N)], axis=0)
    err = np.sqrt((xr - xc) ** 2)

    plt.figure(figsize=(8, 20))
    plt.subplot(311)
    plt.plot(tc, xc, label='Original')
    plt.plot(tc, xr, label='Reconstruct')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal(Original vs Reconstruct)")
    plt.legend()

    plt.subplot(312)
    plt.plot(td, xd, 'r--o')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Sampling Signal")

    plt.subplot(313)
    plt.plot(tc, err, 'm--o')
    plt.xlabel("t(s)")
    plt.ylabel("Err")
    plt.title("Err signal")

    plt.show()


def test6():
    def rect(t, T):
        Th = T / 2.0
        return np.where((t > -Th) & (t < Th), 1.0, 0.0)

    def rect2(t, tao, T):
        N = len(t)
        tao_h = tao / 2.0
        res = np.where((t > -tao_h) & (t < tao_h), 1.0, 0.0)
        k = 1
        while k * T < t[-1]:
            res += np.where((t > k * T - tao_h) & (t < k * T + tao_h), 1.0, 0.0)
            res += np.where((t > -k * T - tao_h) & (t < -k * T + tao_h), 1.0, 0.0)
            k += 1

        return res

    tc = np.arange(-2 * np.pi, 2 * np.pi, 1e-4)
    xc = np.cos(tc) + 1.0

    sc = rect2(tc, 0.5, np.pi / 4.0)
    xs = sc * xc

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(tc, xc, 'b--', label='Origin')
    plt.plot(tc, xs, 'r-', label='RectSam')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.legend(loc='lower left')

    plt.subplot(212)
    # plt.plot(tc, xc, 'r--', label='Original')
    plt.plot(tc, xs, 'b-', label='Reconstruct')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal(Original vs Reconstruct)")
    plt.legend()

    plt.show()


def test7():
    n = np.arange(64)
    x = np.cos(np.pi / 8 * n)
    y = np.cos(np.pi / 8 * n + np.pi / 3)

    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    f = np.arange(64) / 64

    f_shift = f - 1 / 2
    X_shift = np.fft.fftshift(X)
    Y_shift = np.fft.fftshift(Y)

    N_h = 64 // 2
    f_h = f_shift[N_h:]
    X_a = (np.abs(X_shift)[N_h:]) * 2
    Y_a = (np.abs(Y_shift)[N_h:]) * 2

    X_p = np.angle(X_shift)[N_h:]
    Y_p = np.angle(Y_shift)[N_h:]

    plt.figure(figsize=(16, 8))

    plt.subplot(321)
    plt.plot(n, x, 'b-')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.grid(True)

    plt.subplot(322)
    plt.plot(n, y, 'b-')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.grid(True)

    plt.subplot(323)
    plt.plot(f_h, X_a)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(324)
    plt.plot(f_h, Y_a)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(325)
    plt.plot(f_h, X_p)
    plt.xlabel("f(Hz)")
    plt.ylabel("Phase")
    plt.grid(True)

    plt.subplot(326)
    plt.plot(f_h, Y_p)
    plt.xlabel("f(Hz)")
    plt.ylabel("Phase")
    plt.grid(True)

    plt.show()


def test8():
    '''
    np.fft.fft --> X[k]=\sum_{n=0}^{N-1}x[n]e^{-i\frac{2\pi}{N}kn}
    np.fft.ifft --> x[n]=\sum_{k=0}^{N-1}X[k]e^{i\frac{2\pi}{N}kn} / N
    '''
    A = 0.2
    fc = 10
    phase = np.pi / 6.0
    fs = 32 * fc

    t = np.arange(0, 2, 1.0 / fs)  # 2 seconds of sampling time
    x = A * np.sin(2 * np.pi * fc * t + phase)  # 0.2cos(2pi*10t-pi/3)

    N = 256  # N 点傅里叶变换
    X = np.fft.fftshift(np.fft.fft(x, N))
    f = np.arange(N) * fs / N

    freq = f - fs / 2
    # freq = np.fft.fftfreq(t.shape[-1])

    X_no_shift = np.fft.fft(x, N)

    phases = np.angle(X)

    X_backup = np.copy(X)
    threshold = np.max(np.abs(X)) / 1000.0
    X_backup[X_backup < threshold] = 0.0
    phases_cut = np.angle(X_backup)

    x_recon = np.fft.ifft(np.fft.ifftshift(X))
    t_recon = np.arange(0, len(x_recon)) / fs

    plt.figure(figsize=(16, 24))

    plt.subplot(611)
    plt.plot(t, x, 'b-')
    plt.plot(t_recon, np.real(x_recon), 'r-')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(612)
    plt.plot(freq, np.abs(X / N * 2), 'b-')  # A[k]=|X[k]|/N*2
    plt.xlabel("freq(Hz)")
    plt.ylabel("Amplitude")
    plt.title("shift")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(613)
    plt.stem(freq, phases_cut, use_line_collection=True)
    # plt.plot(freq, phases_cut)
    plt.xlabel("freq(Hz)")
    plt.ylabel("Amplitude")
    plt.title("phases cut")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(614)
    plt.plot(t_recon, np.real(x_recon), 'r-')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Reconstruct")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(615)
    plt.plot(freq, np.abs(X_no_shift) / N * 2, 'b-')
    plt.xlabel("freq(Hz)")
    plt.ylabel("Amplitude")
    plt.title("no shift")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(616)
    plt.plot(freq, phases)
    plt.xlabel("freq(Hz)")
    plt.ylabel("Amplitude")
    plt.title("phases no cut")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.show()


def test9():
    fs = 64
    ts = 1 / fs
    t = np.arange(0, 10, ts)
    a_f = [(3, 1, np.pi / 6), (1, 4, np.pi / 2), (1, 32, 0), (2, 3, -np.pi), (0.75, 2, np.pi)]

    N = len(t)
    f = np.arange(N) * fs / N

    x = np.sum([a * np.cos(2 * np.pi * f * t + p) for a, f, p in a_f], axis=0) + np.sqrt(3) / 2
    X = np.fft.fft(x)
    X_shift = np.fft.fftshift(X)
    f_shift = f - fs / 2

    X_backup = np.where(np.abs(X_shift) < 1e-6, 0, X_shift)  # real(x)<0.3

    print("X[{}]={}; {}".format(N / 2, X_shift[0], X_shift[0] * 2 / N))
    print("X[{}]={}; {}".format(0, X_shift[N // 2], X_shift[N // 2] * 2 / N))
    # print("len(f)={}\nf={}".format(len(f_shift), f_shift))

    plt.figure(figsize=(16, 24))

    plt.subplot(611)
    plt.plot(t, x, 'b-')
    # plt.plot(t_recon, np.real(x_recon), 'r-')
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(612)
    # plt.plot(f_shift, np.abs(X_shift) * 2 / N, 'b-')
    plt.stem(f_shift, np.abs(X_shift) * 2 / N, use_line_collection=True)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude spectrum")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(613)
    # plt.stem(f_shift, np.angle(X_shift), use_line_collection=True)
    plt.stem(f_shift, np.angle(X_backup), use_line_collection=True)
    plt.xlabel("f(Hz)")
    plt.ylabel("Phase")
    plt.title("Phase spectrum")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.show()
    print("done")


if __name__ == '__main__':
    test9()
