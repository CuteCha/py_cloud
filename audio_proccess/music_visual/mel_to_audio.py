import numpy as np

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    plt.subplots_adjust(hspace=1)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.subplots_adjust(hspace=1)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


def plot_pitch(waveform, sr, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)


def main():
    filename = "/Users/cxq/code/py_cloud/data/audio_mp3/vad_example.wav"

    time_series, sample_rate = librosa.load(filename)
    stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    y_hat = librosa.feature.inverse.mel_to_audio(M=spectrogram, n_fft=2048 * 4, hop_length=512)

    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    librosa.display.waveshow(time_series, ax=ax[0])
    librosa.display.specshow(spectrogram, y_axis='log', x_axis='time', ax=ax[1])
    librosa.display.waveshow(y_hat, ax=ax[2])
    plt.show()


def run():
    filename = "/Users/cxq/code/py_cloud/data/audio_mp3/vad_example.wav"
    speech_waveform, sample_rate = torchaudio.load(filename)

    n_fft = 1024
    spectrogram = T.Spectrogram(n_fft=n_fft)
    griffin_lim = T.GriffinLim(n_fft=n_fft)
    spec = spectrogram(speech_waveform)
    reconstructed_waveform = griffin_lim(spec)
    reconstructed_spec = spectrogram(reconstructed_waveform)

    # _, axes = plt.subplots(4, 1, sharex=True, sharey=True)
    _, axes = plt.subplots(4, 1)
    plot_waveform(speech_waveform, sample_rate, title="Original wave", ax=axes[0])
    plot_waveform(reconstructed_waveform, sample_rate, title="Reconstructed wave", ax=axes[1])
    plot_spectrogram(spec[0], title="Original spectrogram", ax=axes[2])
    plot_spectrogram(reconstructed_spec[0], title="Reconstructed spectrogram", ax=axes[3])
    plt.show()


def run2():
    filename = "/Users/cxq/code/py_cloud/data/audio_mp3/vad_example.wav"
    speech_waveform, sample_rate = torchaudio.load(filename)
    n_fft = 1024
    spectrogram = T.Spectrogram(n_fft=n_fft)
    griffin_lim = T.GriffinLim(n_fft=n_fft)
    spec = spectrogram(speech_waveform)
    reconstructed_waveform = griffin_lim(spec)
    reconstructed_spec = spectrogram(reconstructed_waveform)

    _, axes = plt.subplots(4, 1)

    speech_waveform_np = speech_waveform.numpy()
    num_channels, num_frames = speech_waveform_np.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    axes[0].plot(time_axis, speech_waveform_np[0], linewidth=1)
    axes[0].grid(True)
    axes[0].set_xlim([0, time_axis[-1]])
    axes[0].set_title("Original wave")
    plt.subplots_adjust(hspace=1)

    axes[1].set_title("Original spectrogram")
    axes[1].set_ylabel("freq_bin")
    axes[1].imshow(librosa.power_to_db(spec[0]), origin="lower", aspect="auto", interpolation="nearest")
    plt.subplots_adjust(hspace=1)

    reconstructed_waveform_np = reconstructed_waveform.numpy()
    num_channels, num_frames = reconstructed_waveform_np.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    axes[2].plot(time_axis, reconstructed_waveform_np[0], linewidth=1)
    axes[2].grid(True)
    axes[2].set_xlim([0, time_axis[-1]])
    axes[2].set_title("Reconstructed wave")
    plt.subplots_adjust(hspace=1)

    axes[3].set_title("Reconstructed spectrogram")
    axes[3].set_ylabel("freq_bin")
    axes[3].imshow(librosa.power_to_db(reconstructed_spec[0]), origin="lower", aspect="auto", interpolation="nearest")
    plt.subplots_adjust(hspace=1)

    plt.show()


def run3():
    n_fft = 256
    n_mels = 64
    sample_rate = 6000

    mel_filters = F.melscale_fbanks(
        int(n_fft // 2 + 1),
        n_mels=n_mels,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        sample_rate=sample_rate,
        norm="slaney",
    )
    plot_fbank(mel_filters, "Mel Filter Bank - torchaudio")

    mel_filters_librosa = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sample_rate / 2.0,
        norm="slaney",
        htk=True,
    ).T

    plot_fbank(mel_filters_librosa, "Mel Filter Bank - librosa")
    mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
    print("Mean Square Difference: ", mse)
    plt.show()


def run4():
    filename = "/Users/cxq/code/py_cloud/data/audio_mp3/vad_example.wav"
    # speech_waveform, sample_rate = librosa.load(filename)
    speech_waveform, sample_rate = torchaudio.load(filename)

    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec_librosa = librosa.feature.melspectrogram(
        y=speech_waveform.numpy()[0],
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=n_mels,
        norm="slaney",
        htk=True,
    )

    melspec = mel_spectrogram(speech_waveform)

    _, axes = plt.subplots(2, 1, figsize=(13, 13))
    plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq", ax=axes[0])
    plot_spectrogram(melspec_librosa, title="MelSpectrogram - librosa", ylabel="mel freq", ax=axes[1])
    mse = torch.square(melspec - melspec_librosa).mean().item()
    print("Mean Square Difference: ", mse)
    plt.show()


def run5():
    filename = "/Users/cxq/code/py_cloud/data/audio_mp3/vad_example.wav"
    speech_waveform, sample_rate = torchaudio.load(filename)

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=256,
        melkwargs={
            "n_fft": 2048,
            "n_mels": 256,
            "hop_length": 512,
            "mel_scale": "htk",
        },
    )
    mfcc = mfcc_transform(speech_waveform)

    lfcc_transform = T.LFCC(
        sample_rate=sample_rate,
        n_lfcc=256,
        speckwargs={
            "n_fft": 2048,
            "win_length": None,
            "hop_length": 512,
        },
    )
    lfcc = lfcc_transform(speech_waveform)

    pitch = F.detect_pitch_frequency(speech_waveform, sample_rate)

    plot_spectrogram(mfcc[0], title="MFCC")
    plot_spectrogram(lfcc[0], title="LFCC")
    plot_pitch(speech_waveform, sample_rate, pitch)
    plt.show()


def debug():
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.cbook as cbook
    import matplotlib.cm as cm
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    fig, ax = plt.subplots(5, 1, figsize=(12, 12))
    ax[0].imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,
                 origin='lower', extent=[-3, 3, -3, 3],
                 vmax=abs(Z).max(), vmin=-abs(Z).max())
    ax[1].plot(x, np.sin(x * 5), label='sin')
    plt.show()


if __name__ == '__main__':
    # run()
    # debug()
    # run2()
    run5()
