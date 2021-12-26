import sys
import argparse
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from vad import VAD

model_name = "container_0/wav2vec2-large-xlsr-ja"
device = "cuda"
processor_name = "container_0/wav2vec2-large-xlsr-ja"

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(processor_name)


def load_file_to_data(file):
    batch = {}
    speech, sampling_rate = torchaudio.load(file)
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
    print(f"{sampling_rate=},{speech.shape=},{speech.squeeze(0).shape=}")
    print(speech.squeeze(0).numpy())
    batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    return batch


def predict(data):
    features = processor(
        data["speech"],
        sampling_rate=data["sampling_rate"],
        padding=True,
        return_tensors="pt",
    )
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    decoded_results = []
    for logit in logits:
        pred_ids = torch.argmax(logit, dim=-1)
        decoded_results.append(processor.decode(pred_ids))
    return decoded_results


def vad_predict():
    vad = VAD(0.8)
    vad.start()
    try:
        while True:
            data = {}
            speech_i16 = np.frombuffer(vad.get_voice(), np.int16)
            data["speech"] = vad._int2float(speech_i16)
            data["sampling_rate"] = 16_000
            out = predict(data)
            print(out)
    except KeyboardInterrupt:
        vad.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav")
    args = parser.parse_args()

    if args.wav is None:
        vad_predict()
    else:
        wav = load_file_to_data(args.wav)
        out = predict(wav)
        print(out)
