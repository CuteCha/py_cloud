# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import librosa
import librosa.display
from scipy.fftpack import dct
from scipy.io import wavfile


def test01():
    fs = 1024
    t1 = np.arange(0, 0.2, 1 / fs)
    t2 = np.arange(0.2, 0.4, 1 / fs)
    t3 = np.arange(0.4, 0.7, 1 / fs)
    t4 = np.arange(0.7, 1.0, 1 / fs)

    y = np.concatenate([np.cos(2 * np.pi * 10 * t1), np.cos(2 * np.pi * 25 * t2),
                        np.cos(2 * np.pi * 50 * t3), np.cos(2 * np.pi * 100 * t4)])

    t = np.concatenate([t1, t2, t3, t4])
    x = np.cos(2 * np.pi * 10 * t) + np.cos(2 * np.pi * 25 * t) \
        + np.cos(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 100 * t)

    n = len(t)
    f = np.arange(n) * fs / n - fs / 2
    xf = np.fft.fftshift(np.fft.fft(x))
    yf = np.fft.fftshift(np.fft.fft(y))

    xfr, xt, xc = signal.stft(x, fs=fs, window='hann', nperseg=256, noverlap=128)
    yfr, yt, yc = signal.stft(y, fs=fs, window='hann', nperseg=256, noverlap=128)

    xa = np.abs(xc)
    ya = np.abs(yc)

    plt.figure(figsize=(16, 8))

    plt.subplot(321)
    plt.plot(t, x)
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("stable wave")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(322)
    plt.plot(t, y)
    plt.xlabel("t(s)")
    plt.ylabel("Amplitude")
    plt.title("unstable wave")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(323)
    plt.plot(f, np.abs(xf) * 2 / n)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("stable spectrum")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(324)
    plt.plot(f, np.abs(yf) * 2 / n)
    plt.xlabel("f(Hz)")
    plt.ylabel("Amplitude")
    plt.title("stable spectrum")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(325)
    plt.pcolormesh(xt, xfr, xa, vmin=0, vmax=xa.mean() * 10)
    plt.title("stft spectrum")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(326)
    plt.pcolormesh(yt, yfr, ya, vmin=0, vmax=ya.mean() * 10)
    plt.title("stft spectrum")
    plt.grid(True)
    plt.subplots_adjust(hspace=0.4)

    plt.show()
    print("done")


def test02():
    duration = 2.0
    Fs = 2000
    omega = 10
    N = int(duration * Fs)
    t = np.arange(N) / Fs
    x = 0.9 * np.sin(2 * np.pi * omega * t * t)

    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.plot(t, x)
    plt.xlim([t[0], t[-1]])
    plt.ylim([-1.1, 1.1])
    plt.xlabel('t(s)')

    plt.subplot(122)
    A = np.abs(np.fft.fft(x)) / N * 2
    freq = np.fft.fftfreq(N, d=1 / Fs)
    n_h = N // 2
    A = A[:n_h]
    freq = freq[:n_h]
    plt.plot(freq, A)
    plt.xlim([0, 50])
    plt.ylim(bottom=0)
    plt.xlabel('f(Hz)')
    plt.tight_layout()

    plt.show()
    print("done")


def windowed_ft(t, x, Fs, w_pos_sec, w_len, w_type, upper_y=1.0):
    N = len(x)
    w_pos = int(Fs * w_pos_sec)
    w = np.zeros(N)
    w[w_pos:w_pos + w_len] = signal.get_window(w_type, w_len)
    x = x * w

    plt.figure(figsize=(8, 2))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, c='k')
    plt.plot(t, w, c='r')
    plt.xlim([min(t), max(t)])
    plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (seconds)')

    plt.subplot(1, 2, 2)
    X = np.abs(np.fft.fft(x)) / N * 2
    freq = np.fft.fftfreq(N, d=1 / Fs)
    X = X[:N // 2]
    freq = freq[:N // 2]
    plt.plot(freq, X, c='k')
    plt.xlim([0, 50])
    plt.ylim([0, upper_y])
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


def test03():
    duration = 2.0
    Fs = 2000
    omega = 10
    N = int(duration * Fs)
    t = np.arange(N) / Fs
    x = 0.9 * np.sin(2 * np.pi * omega * t * t)
    w_len = 1024
    w_pos = 1280
    print('Rectangular window:')
    windowed_ft(t, x, Fs, 1.0, w_len, 'boxcar', upper_y=0.15)
    print('Triangular window:')
    windowed_ft(t, x, Fs, 1.0, w_len, 'triang', upper_y=0.15)
    print('Hann window:')
    windowed_ft(t, x, Fs, 1.0, w_len, 'hann', upper_y=0.15)

    print("done")


def pre_emphasis(signal, coefficient=0.97):
    '''
    对信号进行预加重
    y[n]=x[n]-a*x[n-1]
    '''
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def audio2frame(signal, frame_length, frame_step, winfunc=lambda x: np.ones((x,))):
    '''
    分帧
    '''
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    if signal_length <= frame_length:
        frames_num = 1
    else:
        frames_num = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)
    zeros = np.zeros((pad_length - signal_length,))
    pad_signal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1)) + np.tile(
        np.arange(0, frames_num * frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    win = np.tile(winfunc(frame_length), (frames_num, 1))

    return frames * win


def deframesignal(frames, signal_length, frame_length, frame_step, winfunc=lambda x: np.ones((x,))):
    '''加窗'''
    signal_length = round(signal_length)
    frame_length = round(frame_length)
    frames_num = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_length, '"frames"矩阵大小不正确，它的列数应该等于一帧长度'
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1)) + np.tile(
        np.arange(0, frames_num * frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)
    pad_length = (frames_num - 1) * frame_step + frame_length
    if signal_length <= 0:
        signal_length = pad_length
    recalc_signal = np.zeros((pad_length,))
    window_correction = np.zeros((pad_length, 1))
    win = winfunc(frame_length)
    for i in range(0, frames_num):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + win + 1e-15
        recalc_signal[indices[i, :]] = recalc_signal[indices[i, :]] + frames[i, :]
    recalc_signal = recalc_signal / window_correction
    return recalc_signal[0:signal_length]


def hz2mel(hz):
    '''
    把频率hz转化为梅尔频率
    '''

    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    '''
    把梅尔频率转化为hz
    '''

    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filter_banks(filters_num=20, NFFT=512, samplerate=16000, low_freq=0, high_freq=None):
    '''
    计算梅尔三角间距滤波器，该滤波器在第一个频率和第三个频率处为0，在第二个频率处为1
    '''
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, filters_num + 2)
    hz_points = mel2hz(mel_points)
    bin = np.floor((NFFT + 1) * hz_points / samplerate)
    fbank = np.zeros([filters_num, NFFT / 2 + 1])
    for j in range(0, filters_num):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])

    return fbank


def lifter(cepstra, L=22):
    '''
    升倒谱函数
    '''
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        return cepstra


def derivate(feat, big_theta=2, cep_num=13):
    '''
    计算一阶系数或者加速系数的一般变换公式
    '''
    result = np.zeros(feat.shape)
    denominator = 0
    for theta in np.linspace(1, big_theta, big_theta):
        denominator = denominator + theta ** 2
    denominator = denominator * 2
    for row in np.linspace(0, feat.shape[0] - 1, feat.shape[0]):
        tmp = np.zeros((cep_num,))
        numerator = np.zeros((cep_num,))
        for t in np.linspace(1, cep_num, cep_num):
            a = 0
            b = 0
            s = 0
            for theta in np.linspace(1, big_theta, big_theta):
                if (t + theta) > cep_num:
                    a = 0
                else:
                    a = feat[row][t + theta - 1]
                if (t - theta) < 1:
                    b = 0
                else:
                    b = feat[row][t - theta - 1]
                s += theta * (a - b)
            numerator[t - 1] = s
        tmp = numerator * 1.0 / denominator
        result[row] = tmp
    return result


def mfcc_extract():
    audio_file = "/data/workflow/data/UrbanSound8K/audio/fold1/101415-3-0-2.wav"
    signal, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr=sr)
    delta1_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    comp_mfccs = np.concatenate([mfccs, delta1_mfccs, delta2_mfccs])

    print(mfccs.shape, delta1_mfccs.shape, delta2_mfccs.shape, comp_mfccs.shape)

    plt.figure(figsize=(10, 30))

    plt.subplot(411)
    librosa.display.specshow(mfccs, x_axis="time", sr=sr)
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(412)
    librosa.display.specshow(delta1_mfccs, x_axis="time", sr=sr)
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(413)
    librosa.display.specshow(delta2_mfccs, x_axis="time", sr=sr)
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(414)
    librosa.display.waveshow(signal, sr=sr)

    plt.show()
    print("done")

    # np.fft.rfft() = np.fft.fft()[:(N//2+1)]


'''
def stftAnal(x, w, N, H):
    """
    x: 输入信号, w: 分析窗, N: FFT 的大小, H: hop 的大小
    返回 xmX, xpX: 振幅和相位，以 dB 为单位
    """

    M = w.size  # 分析窗的大小
    hM1 = (M + 1) // 2
    hM2 = M // 2
    x = np.append(np.zeros(hM2), x)  # 在信号 x 的最前面与最后面补零
    x = np.append(x, np.zeros(hM2))
    pin = hM1  # 初始化指针，用来指示现在指示现在正在处理哪一帧
    pend = x.size - hM1  # 最后一帧的位置
    w = w / sum(w)  # 归一化分析窗
    xmX = []
    xpX = []
    while pin <= pend:
        x1 = x[pin - hM1:pin + hM2]  # 选择一帧输入的信号
        mX, pX = dftAnal(x1, w, N)  # 计算 DFT（这个函数不是库中的）
        xmX.append(np.array(mX))  # 添加到 list 中
        xpX.append(np.array(pX))
        pin += H  # 更新指针指示的位置
    xmX = np.array(xmX)  # 转换为 numpy 数组
    xpX = np.array(xpX)

    return xmX, xpX


def stftSynth(mY, pY, M, H):
    """
    mY: 振幅谱以dB为单位, pY: 相位谱, M: 分析窗的大小, H: hop 的大小
    返回 y 还原后的信号
    """
    hM1 = (M + 1) // 2
    hM2 = M // 2
    nFrames = mY[:, 0].size  # 计算帧的数量
    y = np.zeros(nFrames * H + hM1 + hM2)  # 初始化输出向量
    pin = hM1
    for i in range(nFrames):  # 迭代所有帧
        y1 = dftSynth(mY[i, :], pY[i, :], M)  # 计算IDFT（这个函数不是库中的）
        y[pin - hM1:pin + hM2] += H * y1  # overlap-add
        pin += H  # pin是一个指针，用来指示现在指示现在正在处理哪一帧
    y = np.delete(y, range(hM2))  # 删除头部在stftAnal中添加的部分
    y = np.delete(y, range(y.size - hM1, y.size))  # 删除尾部在stftAnal中添加的部分

    return y
'''

if __name__ == '__main__':
    mfcc_extract()
