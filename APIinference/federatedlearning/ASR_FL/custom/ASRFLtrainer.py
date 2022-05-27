from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from packaging import version
import datasets
import torch
import os
import torch.nn as nn
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import get_data
device = torch.device("cpu")

if version.parse(torch.__version__) >= version.parse("1.6"):

    _is_native_amp_available = True


def get_processor():
    print(os.getcwd(), flush=True)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16_000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
    tokenizer = Wav2Vec2CTCTokenizer("/home/VICOMTECH/igonzalez/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def get_model():
    
    processor = get_processor()
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model = model.to(device)
    return(model)


def compute_metrics(pred):
    processor = get_processor()
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_metric = datasets.load_metric("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


class FLTrainer(Executor):
    def __init__(self,
                 lr=0.01,
                 epochs=1,
                 train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
                 validate_task_name=AppConstants.TASK_VALIDATION,
                 exclude_vars=None):
        super(FLTrainer, self).__init__()
        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._validate_task_name = validate_task_name
        self._exclude_vars = exclude_vars

        # Training setup
        #self.model = SimpleNetwork()
        self.model = get_model()
        self.processor = get_processor()
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model.to(self.device)
        #self.loss = nn.CrossEntropyLoss()
        #self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.train_dataset, self.eval_dataset = get_data.train_test_dataset()

        self._train_loader = None  # leave them none until first time they are loaded
        self._test_loader = None
        self._n_iterations = None  # to be set later

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)
        print("INIT DONE -------------------------------------------------", flush=True)


    def local_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        print("LOCAL TRAIN -------------------------------------------------", flush=True)
        self.model.load_state_dict(state_dict=weights)
        # self._train_loader = self.get_data_loader(fl_ctx=fl_ctx, is_for_training=True) if self._train_loader is None else self._train_loader

        data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        args = TrainingArguments(
            output_dir="",
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            no_cuda=True,
            max_steps=10,
            eval_steps=250,
            logging_dir="",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
        )
        trainer = CTCTrainer(
            model=self.model,
            args=args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.processor.feature_extractor,
        )

        checkpoint = None
        print("PREPARE TRAINING DONE -------------------------------------------------", flush=True)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print("TRAINING DONE -------------------------------------------------", flush=True)
        print(train_result, flush=True)


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            print("EXECUTE -------------------------------------------------", flush=True)
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()

            elif task_name == self._validate_task_name:
                self.log_info(fl_ctx, f'Starting task for model validation')
                model_owner = "?"
                try:
                    try:
                        dxo = from_shareable(shareable)
                    except Exception as ex:
                        self.log_error(fl_ctx, f"Error in extracting dxo from shareable. Error: {ex}")
                        return make_reply(ReturnCode.BAD_TASK_DATA)

                    # Ensure data_kind is weights.
                    if not dxo.data_kind == DataKind.WEIGHTS:
                        self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                        return make_reply(ReturnCode.BAD_TASK_DATA)

                    # Extract weights and ensure they are tensor.
                    model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                    weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                    # Get validation accuracy
                    micro_fscore = self.do_validation(weights, abort_signal, fl_ctx)
                    if abort_signal.triggered:
                        return make_reply(ReturnCode.TASK_ABORTED)

                    self.log_info(fl_ctx, f"micro_fscore when validating {model_owner}'s model on"
                                          f" {fl_ctx.get_identity_name()}"f's data: {micro_fscore}')

                    dxo = DXO(data_kind=DataKind.METRICS, data={'micro_fscore': micro_fscore})
                    return dxo.to_shareable()
                except Exception as ex:
                    self.log_exception(fl_ctx, f"Exception in validating model from {model_owner} --> {ex}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)


            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml





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
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.detach()


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
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
        # repace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch