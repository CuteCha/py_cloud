# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from scipy.fftpack import dct
from scipy.io import wavfile


# 绘制时域图
def plot_time(signal, sample_rate):
    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()


# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate // 2, fft_size // 2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()


# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()


def load_wave(wave_file):
    sample_rate, signal = wavfile.read('/data/workflow/data/OSR_us_000_0010_8k.wav')
    signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    print('sample rate:{}; signal length: {}'.format(sample_rate, len(signal)))

    return sample_rate, signal


def emp_signal(signal, alpha=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal


def seg_frames(emphasized_signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    print("frame_length: {}; frame_step: {}".format(frame_length, frame_step))

    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
    print("emphasized_signal_length: {}, num_frames: {}".format(signal_length, num_frames))

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    ids = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1, 1)
    frames = pad_signal[ids]
    print("frames.shape: {}".format(frames.shape))

    return frames, frame_length, frame_step


def window_frame(frames, frame_length):
    hamming = np.hamming(frame_length)
    return frames * hamming


def fft_frame(frames, n_fft=512):
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))
    print(pow_frames.shape)

    return pow_frames


def gen_mel_filter(sample_rate, n_filter, n_fft, low_mel, high_mel):
    mel_points = np.linspace(low_mel, high_mel, n_filter + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((n_filter, int(n_fft / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (n_fft / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, n_filter + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])

    print(fbank.shape)
    # print(fbank)

    return fbank


def cal_log_mel(pow_frames, fbank):
    mel_specs = np.dot(pow_frames, fbank.T)
    mel_specs = np.where(mel_specs == 0, np.finfo(float).eps, mel_specs)
    log_mel_specs = 20 * np.log10(mel_specs)  # dB
    print(log_mel_specs.shape)

    return log_mel_specs


def cal_mfcc(log_mel_specs):
    num_cep = 12
    mfcc = dct(log_mel_specs, type=2, axis=1, norm='ortho')[:, 1:(num_cep + 1)]
    print(mfcc.shape)

    return mfcc


def mfcc_cal():
    wave_file = '/data/workflow/data/OSR_us_000_0010_8k.wav'
    n_fft = 512

    sample_rate, signal = load_wave(wave_file)
    emphasized_signal = emp_signal(signal)
    frames, frame_length, frame_step = seg_frames(emphasized_signal, sample_rate)
    win_frames = window_frame(frames, frame_length)
    pow_frames = fft_frame(win_frames, n_fft)

    f_max = sample_rate / 2
    n_filter = 26
    low_mel = 0
    high_mel = 2595 * np.log10(1 + f_max / 700)
    f_bank = gen_mel_filter(sample_rate, n_filter, n_fft, low_mel, high_mel)

    log_mel = cal_log_mel(pow_frames, f_bank)
    mfcc = cal_mfcc(log_mel)

    print(mfcc.shape)
    plot_spectrogram(mfcc.T, 'MFCC Coefficients')
    print("done")


if __name__ == '__main__':
    mfcc_cal()
