from funasr import AutoModel
import json


def asr_streaming():
    chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

    model = AutoModel(model="paraformer-zh-streaming")

    import soundfile
    import os

    print(model.model_path)
    wav_file = os.path.join(model.model_path, "example/asr_example.wav")
    speech, sample_rate = soundfile.read(wav_file)
    chunk_stride = chunk_size[1] * 960  # 600ms

    cache = {}
    total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
                             encoder_chunk_look_back=encoder_chunk_look_back,
                             decoder_chunk_look_back=decoder_chunk_look_back)
        print(res)


def asr_non_streaming():
    # model = AutoModel(model="fsmn-vad")
    # wav_file = f"{model.model_path}/example/vad_example.wav"
    # res = model.generate(input=wav_file)
    # print(res)

    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )
    res = model.generate(input="/Users/cxq/code/py_cloud/data/audio_mp3/vad_example.wav",
                         batch_size_s=300,
                         hotword='魔搭')
    print(json.dumps(res, ensure_ascii=False))


def punctuation_restoration():
    from funasr_onnx import CT_Transformer

    model_dir = "/Users/cxq/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    model = CT_Transformer(model_dir)

    text_in = "跨境河流是养育沿岸人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切愿意进一步完善双方联合工作机制凡是中方能做的我们都会去做而且会做得更好我请印度朋友们放心中国在上游的任何开发利用都会经过科学规划和论证兼顾上下游的利益"
    result = model(text_in)
    print(result[0])
    print(len(result[0]))
    print(len(result[1]))
    print(json.dumps(result, ensure_ascii=False))


def main():
    model = AutoModel(model="paraformer-zh")  # ~/.cache/modelscope/hub/iic/
    res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav")
    # print(res)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    # main()
    asr_non_streaming()
    # asr_streaming()
    # punctuation_restoration()
