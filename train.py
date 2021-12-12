import argparse
import yaml
import json
from pathlib import Path
from datasets import Dataset, load_metric
from pprint import pprint
import torchaudio

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer

from functools import partial
from data_collator_ctc_with_padding import DataCollatorCTCWithPadding


def create_dataset(yaml_file, wav_dir: Path):
    paths = []
    texts = []

    with open(yaml_file) as f:
        meta = yaml.safe_load(f)

    for i, m in enumerate(meta.keys()):
        wav = wav_dir / m
        paths.append(f"{wav}.wav")

        sentence = meta[m]["phone_level3"]
        sentence = sentence.replace("-pau", " ")
        sentence = sentence.replace("-", "")
        sentence = sentence.lower() + " "
        texts.append(sentence)
    data = Dataset.from_dict({"path": paths, "text": texts})
    data = data.shuffle(seed=42)
    return data.train_test_split(test_size=0.1, seed=42)


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    # print(f"{all_text=}")
    # print(f"{vocab=}")
    return {"vocab": [vocab], "all_text": [all_text]}


def speech_file_to_array(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
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


def compute_metrics(processor, pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


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

    vocab_train = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_dataset.column_names,
    )

    vocab_train = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_dataset.column_names,
    )
    vocab_test = test_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=test_dataset.column_names,
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open("vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
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

    processor.save_pretrained("container_0/wav2vec2-large-xlsr-ja")

    train_dataset = train_dataset.map(
        speech_file_to_array, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        speech_file_to_array, remove_columns=test_dataset.column_names
    )

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
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="container_0/ckpts/",
        logging_dir="container_0/runs/",
        group_by_length=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=200,
        eval_steps=200,
        logging_steps=200,
        learning_rate=4e-4,
        warmup_steps=int(0.1 * 1320),  # 10%
        save_total_limit=2,
    )

    _compute_metrics = partial(compute_metrics, processor)

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
    trainer.save_model("container_0/wav2vec2-large-xlsr-kn")


if __name__ == "__main__":
    main()
