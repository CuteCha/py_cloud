import whisper
import json


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


def cn_debug():
    audio_path = "/Users/cxq/code/py_cloud/data/audio_mp3/marss.wav"

    model = whisper.load_model("base")  # ~/.cache/whisper/
    result = model.transcribe(audio_path)
    seg_result = model.transcribe(audio_path,
                                  initial_prompt="加上标点符号；如果使用了中文，请使用简体中文来表示文本内容。")
    print(result["text"])
    print(json.dumps(seg_result, ensure_ascii=False))

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    # options = whisper.DecodingOptions(beam_size=5)
    # seg_res = whisper.decode(model, mel, options)
    # print(json.dumps(seg_res, ensure_ascii=False))


if __name__ == '__main__':
    # main()
    cn_debug()
