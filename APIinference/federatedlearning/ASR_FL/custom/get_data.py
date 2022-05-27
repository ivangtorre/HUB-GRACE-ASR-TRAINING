import datasets
from datasets import Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

def train_test_dataset():
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    # THIS IS HARDCODED AND SHOULD BE CONFIGURED IN THE FUTURE
    # TAMBIEN SE HA ELIMINADO CUANDO SE GENERAL EL AUDIO
    train_dataset = datasets.load_dataset("common_voice", "ga-IE", split="train[:10%]")
    eval_dataset = datasets.load_dataset("common_voice", "ga-IE", split="test[:3%]")

    train_dataset = train_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    eval_dataset = eval_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])


    # Create and save tokenizer
    import re
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

    def remove_special_characters(batch):
        batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
        return batch

    train_dataset = train_dataset.map(remove_special_characters)
    eval_dataset = eval_dataset.map(remove_special_characters)

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)



    return train_dataset, eval_dataset

