import argparse
import yaml
import json
from pathlib import Path
from datasets import Dataset, load_metric
from pprint import pprint
import librosa
import numpy as np
import torchaudio

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer

from functools import partial
from data_collator_ctc_with_padding import DataCollatorCTCWithPadding

from j2a import j2a

def create_dataset(tsv, wav: Path):
    paths = []
    texts = []

    txt_list = j2a(tsv)

    for i, t in enumerate(txt_list):
        paths.append(f"{i}.wav")

        texts.append(t[2])
    data = Dataset.from_dict({"path": paths, "text": texts})
    data = data.shuffle(seed=42)
    return data.train_test_split(test_size=0.001, seed=42)


def speech_file_to_array(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch


def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch


def prepare_dataset(processor, batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


def compute_metrics(processor, wer_metric, pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# https://discuss.huggingface.co/t/wav2vec2-0-memory-issue/4868
def remove_long_common_voicedata(dataset, max_seconds=6):
    # convert pyarrow table to pandas
    dftest = dataset.to_pandas()

    # find out length of input_values
    dftest["len"] = dftest["speech"].apply(len)

    # for wav2vec training we already resampled to 16khz
    # remove data that is longer than max_seconds (6 seconds ideal)
    maxLength = max_seconds * 16000
    dftest = dftest[dftest["len"] < maxLength]
    dftest = dftest.drop("len", 1)

    # convert back to pyarrow table to use in trainer
    return dataset.from_pandas(dftest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        required=True,
    )
    parser.add_argument(
        "--wav-dir",
        required=True,
    )
    args = parser.parse_args()

    dataset = create_dataset(args.yaml, Path(args.wav_dir))
    pprint(dataset)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = Wav2Vec2CTCTokenizer(
        "./container_1/wav2vec2-large-xlsr-ja/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    processor.save_pretrained("container_1/wav2vec2-large-xlsr-ja")

    train_dataset = train_dataset.map(speech_file_to_array)
    test_dataset = test_dataset.map(speech_file_to_array)

    train_dataset = remove_long_common_voicedata(train_dataset)
    test_dataset = remove_long_common_voicedata(test_dataset)

    print(train_dataset["sampling_rate"][:10])

    _prepare_dataset = partial(prepare_dataset, processor)

    train_dataset = train_dataset.map(
        _prepare_dataset,
        remove_columns=train_dataset.column_names,
        batch_size=4,
        num_proc=4,
        batched=True,
    )
    test_dataset = test_dataset.map(
        _prepare_dataset,
        remove_columns=test_dataset.column_names,
        batch_size=4,
        num_proc=4,
        batched=True,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "container_1/wav2vec2-large-xlsr-ja",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        # gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="container_1/ckpts/",
        logging_dir="container_1/runs/",
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=1000,
        learning_rate=4e-4,
        warmup_steps=int(0.1 * len(train_dataset)),  # 10%
        save_total_limit=2,
    )

    _compute_metrics = partial(compute_metrics, processor, wer_metric)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model("container_1/wav2vec2-large-xlsr-ja")


if __name__ == "__main__":
    main()
