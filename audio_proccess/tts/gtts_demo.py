import os
from gtts import gTTS
from playsound import playsound


def main():
    os.environ["https_proxy"] = "http://127.0.0.1:1080"

    # 谷歌文字转语音API测试
    text = "测试gtts文本转语音"
    audio = gTTS(text=text, lang="zh-cn")
    print(os.getcwd())
    print(os.path.dirname(os.path.dirname(os.getcwd())))
    print(audio)
    path_mp3 = f"{os.path.dirname(os.path.dirname(os.getcwd()))}/data/demo.mp3"
    audio.save(path_mp3)
    playsound(path_mp3)


if __name__ == '__main__':
    main()
