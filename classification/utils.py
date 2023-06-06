# -*- encoding:utf-8 -*-

import librosa
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.simplefilter("ignore", category=FutureWarning)


def extract_features(audio_file):
    wav, sr = librosa.load(audio_file)
    amps = np.abs(librosa.stft(wav))
    # print("amps.shape: {}".format(amps.shape))
    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20).T
    chroma = librosa.feature.chroma_stft(S=amps, sr=sr).T
    mel = librosa.feature.melspectrogram(wav, sr=sr).T
    contrast = librosa.feature.spectral_contrast(S=amps, sr=sr).T
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(wav), sr=sr).T

    return mfccs, chroma, mel, contrast, tonnetz


def get_mfcc_mean(audio_file):
    wav, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40).T  # (num_seq,n_mfcc)

    return np.mean(mfccs, axis=0)  # (n_mfcc, )


def one_hot_encode(class_id):
    label = np.zeros(10)
    label[class_id] = 1

    return label


def get_features(audio_file):
    mfccs, chroma, mel, contrast, tonnetz = extract_features(audio_file)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return np.mean(ext_features, axis=0)


def gen_classification_samples0():
    audio_dirs = "/data/workflow/data/UrbanSound8K/audio/"
    audio_instruction = "/data/workflow/data/UrbanSound8K/metadata/UrbanSound8K.csv"
    samples_file = "/data/workflow/data/UrbanSound8K/samples/dnn/samples.npy"

    instruction = pd.read_csv(audio_instruction)
    samples = [[get_features("{}fold{}/{}".format(audio_dirs, row.fold, row.slice_file_name)), row.classID]
               for row in instruction.itertuples()]
    # cnt = 0
    # for row in instruction.itertuples():
    #     cnt += 1
    #     if cnt > 3: break
    #     class_id = row.classID
    #     audio_file = "{}fold{}/{}".format(audio_dirs, row.fold, row.slice_file_name)
    #     features = get_features(audio_file)
    #     samples.append([features, class_id])
    #     print("class_id: {}; features.shape: {}; audio_file: {}".format(class_id, features.shape, audio_file))

    np.save(samples_file, np.array(samples, dtype=object))


def gen_classification_samples():
    audio_dirs = "/data/workflow/data/UrbanSound8K/audio/"
    audio_instruction = "/data/workflow/data/UrbanSound8K/metadata/UrbanSound8K.csv"
    samples_file = "/data/workflow/data/UrbanSound8K/samples/dnn/samples.npy"

    instruction = pd.read_csv(audio_instruction)
    samples = []
    cnt = 0
    for row in instruction.itertuples():
        cnt += 1
        print(cnt, flush=True)
        class_id = row.classID
        audio_file = "{}fold{}/{}".format(audio_dirs, row.fold, row.slice_file_name)
        features = get_features(audio_file)
        samples.append([features, class_id])

    np.save(samples_file, np.array(samples, dtype=object))


def load_samples():
    samples_file = "/data/workflow/data/UrbanSound8K/samples/dnn/samples.npy"
    data = np.load(samples_file, allow_pickle=True)
    print((data[0, 0]).shape, data[0, 1])
    print(data.shape)


def main():
    gen_classification_samples()
    # load_samples()


if __name__ == '__main__':
    main()
