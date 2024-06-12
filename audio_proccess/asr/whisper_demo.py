import whisper


def main():
    # audio_path = "/Users/cxq/code/py_cloud/data/audio_mp3/h1.mp3"
    audio_path = "/Users/cxq/code/py_cloud/data/audio_mp3/en_test.mp3"
    # audio_path = "/Users/cxq/code/py_cloud/data/audio_mp3/hokkien_test2.mp3"
    model = whisper.load_model("base")  # ~/.cache/whisper/
    result = model.transcribe(audio_path)
    print(result["text"])

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")


if __name__ == '__main__':
    main()
