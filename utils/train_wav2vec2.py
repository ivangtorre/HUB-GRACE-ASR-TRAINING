#!/usr/bin/env python3
import json
import logging
import os
os.environ['TRANSFORMERS_CACHE'] = "/DATA/cache"
import pandas as pd
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import librosa
import random
import datasets
import numpy as np
import torch
from packaging import version
from torch import nn
from datasets import load_dataset, Dataset, load_from_disk
from apex import amp

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process
transformers.logging.set_verbosity_info()

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    cache_dir: Optional[str] = field(default=True, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},)
    model_cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},)
    freeze_feature_extractor: Optional[bool] = field(default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."})
    attention_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."})
    activation_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout ratio for activations inside the fully connected layer."})
    hidden_dropout: Optional[float] = field(default=0.1,metadata={"help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."},)
    feat_proj_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout probabilitiy for all 1D convolutional layers in feature extractor."},)
    mask_time_prob: Optional[float] = field(default=0.2,metadata={
            "help": "Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                    "vectors will be masked along the time axis. This is only relevant if ``apply_spec_augment is True``."},)
#    gradient_checkpointing: Optional[bool] = field(default=False,metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."},)
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    lang: Optional[str] = field(default=None, metadata={"help": "The language in case of use hugging face or similar"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    dataset_eval: Optional[str] = field(default=None, metadata={"help": "Eval dataset."})
    train_split_name: Optional[str] = field(default="train+validation", metadata={ "help": "training data set split to use (via the datasets library). Defaults to 'train'"},)
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."})
    preprocessing_num_workers: Optional[int] = field(default=None,metadata={"help": "The number of processes to use for the preprocessing."},)
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes"},)
    max_val_samples: Optional[int] = field(default=None,metadata=dict(help="For debugging purposes"),)
    chars_to_ignore: List[str] = list_field(default=[",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", "^", "/", "$"],
        metadata={"help": "A list of characters to remove from the transcripts."},)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

#updates = 0

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        # UNFREEZE FEATURE UPDATES (HARDCODED)
        #global updates
        #updates += 1
        #if updates == 128*2000: # TODO Cambiar esto para que sean variables
        #    model.wav2vec2.feature_extractor.trainable = True

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)

        else:
            loss = self.compute_loss(model, inputs)
            #loss = model(**inputs).loss

        #loss = model(**inputs).loss
        #if not loss < 100: # Check exploding loss
        #    print(loss)
        #    print(inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")


        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        #loss.backward()
        return loss.detach()



def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,  let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    assert(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True

    import os
    #local_rank = int(os.environ["LOCAL_RANK"])
    #print(local_rank)

    # set up distributed training
    #if local_rank is not None:
    #    torch.cuda.set_device(local_rank)
    #    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    #multi_gpu = torch.distributed.is_initialized()
    #if multi_gpu:
    #    print("DISTRIBUTED TRAINING with {} gpus".format(torch.distributed.get_world_size()))

    # define amp optimiation level
    optim_level = 1# if args.amp else 0



    # Detecting last checkpoint.
    generate_vocab = False
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    elif os.path.isdir(model_args.model_name_or_path):
        generate_vocab=False
        copystring= "cp " + model_args.model_name_or_path + "/vocab.json " + training_args.output_dir + "/vocab.json"
        os.system(copystring)

    else:
        # GENERATE VOCAB
        generate_vocab = True


    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)],)
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    random.seed(3)

#################################################################################################################
    # Get the datasets:
    try:
        df_train = pd.read_csv(data_args.dataset_config_name, delimiter=',')
        df_test = pd.read_csv(data_args.dataset_eval, delimiter=',')

        df_train = df_train[["transcription", "file_path"]]
        df_train.columns = ["sentence", "path"]

        df_train = df_train[~df_train["transcription"].isnull()]
        df_test = df_test[~df_test["transcription"].isnull()]

        df_train = df_train[df_train["duration"] < 20]
        df_test = df_test[df_test["duration"] < 20]
        df_train = df_train[df_train["duration"] > 0.5]
        df_test = df_test[df_test["duration"] > 0.5]

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        df_train = df_train[["transcription", "file_path"]]
        df_train.columns = ["sentence", "path"]

        df_test = df_test[["transcription", "file_path"]]
        df_test.columns = ["sentence", "path"]

        train_dataset = Dataset.from_pandas(df_train)
        eval_dataset = Dataset.from_pandas(df_test)

        del df_train, df_test
        print("################################################\n")
        print("# TOTAL TRAIN TIME DURATION: " + str(round(df_train.duration.sum() / 60 / 60, 2)) + " HOURS")
        print("# TOTAL TEST TIME DURATION: " + str(round(df_test.duration.sum() / 60 / 60, 2)) + " HOURS")
        print("\n################################################\n")

    except:
        print("dataset not found in:" + data_args.dataset_config_name + "or " + data_args.dataset_eval)
        print("Trying to download it")
        train_dataset = datasets.load_dataset(data_args.dataset_config_name, data_args.lang, split="train", cache_dir="/DATA/cache")
        eval_dataset = datasets.load_dataset(data_args.dataset_eval, data_args.lang, split="test", cache_dir="/DATA/cache")
        train_dataset = train_dataset.rename_column("text", "sentence")
        train_dataset = train_dataset.rename_column("audio", "file_path")
        eval_dataset = eval_dataset.rename_column("text", "sentence")
        eval_dataset = eval_dataset.rename_column("audio", "file_path")



    train_dataset.save_to_disk("/DATA/cache/trainset" + str(training_args.local_rank))
    eval_dataset.save_to_disk("/DATA/cache/evalset" + str(training_args.local_rank))

    train_dataset=load_from_disk("/DATA/cache/trainset"+ str(training_args.local_rank))
    eval_dataset=load_from_disk("/DATA/cache/evalset"+ str(training_args.local_rank))


    #################################################################################################################

    if generate_vocab==True:
        print("GENERATING VOCAB...")
        vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=False,
                                        remove_columns=train_dataset.column_names)

        vocab_list = list(set(vocab_train["vocab"][0]))
        vocab_list.insert(0, "[UNK]")
        vocab_list.insert(0, "[PAD]")
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        with open(training_args.output_dir + "/vocab.json", "w") as vocab_file:
            json.dump(vocab_dict, vocab_file)


    # Load pretrained model and tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(training_args.output_dir + "/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        activation_dropout=model_args.activation_dropout,
        attention_dropout=model_args.attention_dropout,
        hidden_dropout=model_args.hidden_dropout,
        feat_proj_dropout=model_args.feat_proj_dropout,
        mask_time_prob=model_args.mask_time_prob,
#        mask_feature_prob=model_args.mask_time_prob,
#        gradient_checkpointing=model_args.gradient_checkpointing,
        gradient_checkpointing=False,
        layerdrop=model_args.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=True
    )

    if data_args.max_train_samples is not (None or 0):
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if data_args.max_val_samples is not (None or 0):
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    def speech_file_to_array_fn(batch):
        batch["speech"], _ = librosa.load(batch["path"], sr=None, mono=False)
        batch["sampling_rate"] = 16_000
        batch["target_text"] = batch["sentence"]
        return batch

    # PREPARE AUDIOS ##
    logger.info("LOADING AUDIOS")
    train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names,
                                      keep_in_memory=False, load_from_cache_file=True, num_proc=40)
    eval_dataset = eval_dataset.map(speech_file_to_array_fn, remove_columns=eval_dataset.column_names,
                                    keep_in_memory=False, load_from_cache_file=True, num_proc=40)

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        #batch["input_values"] = batch["input_values1"]
        # Setup the processor for targets
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    logger.info("\nJUST BEFORE TRAINING")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,
                                      batch_size=training_args.per_device_train_batch_size, batched=True,
                                      keep_in_memory=False, load_from_cache_file=True, num_proc=40)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names,
                                    batch_size=training_args.per_device_train_batch_size, batched=True,
                                    keep_in_memory=False, load_from_cache_file=True, num_proc=40)

    def compute_metrics(pred):
        wer_metric = datasets.load_metric("wer")
        cer_metric = datasets.load_metric("cer")
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        logger.info("")
        logger.info("***** REFERENCE (EVALUATION): ****")
        logger.info(label_str[0:100])
        logger.info("")
        logger.info(" ***** PREDICTION (EVALUATION): *****")
        logger.info(pred_str[0:100])
        logger.info("")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}


    model.freeze_feature_extractor()

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
    )


    # START TRAINING ##---------------------------------------------
    logger.info("\nSTART TRAINING")
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        #train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()

        # save the feature_extractor and the tokenizer
        if is_main_process(training_args.local_rank):
            processor.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # DO EVALUATION ##---------------------------------------------------
    results = {}
    if training_args.do_eval:
        logger.info("******** Evaluate ********")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    return results


if __name__ == "__main__":
    main()
