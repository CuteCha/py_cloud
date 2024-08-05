def tts():
    from comtypes.client import CreateObject
    engine = CreateObject("SAPI.SpVoice")
    stream = CreateObject("SAPI.SpFileStream")
    from comtypes.gen import SpeechLib

    data_path = "/Users/cxq/code/py_cloud/data/audio_mp3"
    infile = f"{data_path}/text.txt"
    outfile = f"{data_path}/speech.wav"
    stream.Open(outfile, SpeechLib.SSFMCreateForWrite)
    engine.AudioOutputStream = stream
    f = open(infile, "r", encoding="utf-8")
    theText = f.read()
    f.close()
    engine.speak(theText)
    stream.close()


def asr():
    import speech_recognition as sr
    r = sr.Recognizer()
    data_path = "/Users/cxq/code/py_cloud/data/audio_mp3"
    audio_file = sr.AudioFile(f"{data_path}/h1.wav")
    with audio_file as source:
        audio_data = r.record(source)

    print(f"content: {r.recognize_sphinx(audio_data, language='zh-CN')}")


if __name__ == '__main__':
    # tts()
    asr()
