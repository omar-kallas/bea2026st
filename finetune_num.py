import logging
import inspect
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from utils import (
    cleanup_trainer_memory,
    compute_metrics,
    load_data_paths,
    load_model_params,
    merge_cols,
)

# Fixed paths
DATA_DIR = Path("data_enriched/")
MODELS_DIR = Path("models/")


class NumericRegressionModel(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=config,
        )
        self.dropout = nn.Dropout(float(getattr(config, "hidden_dropout_prob", 0.1)))
        hidden_size = int(config.hidden_size)
        numeric_size = int(config.num_numeric_features)
        ffn_hidden_size = max(128, hidden_size // 2)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size + numeric_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_hidden_size, 1),
        )
        self.loss_fn = nn.MSELoss()
        self.is_base_frozen = False
        self._encoder_forward_args = set(inspect.signature(self.encoder.forward).parameters.keys())

        if int(getattr(config, "freeze_base_epochs", 0)) > 0:
            self.freeze_base_model()

    def freeze_base_model(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.is_base_frozen = True

    def unfreeze_base_model(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.is_base_frozen = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        numeric_features=None,
        labels=None,
        **kwargs,
    ):
        encoder_kwargs = {k: v for k, v in kwargs.items() if k in self._encoder_forward_args}
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_kwargs,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        numeric_features = numeric_features.to(cls_embedding.dtype)
        fused = torch.cat([cls_embedding, numeric_features], dim=-1)
        logits = self.ffn(self.dropout(fused))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1), labels.view(-1).float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NumericDataCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        numeric_features = torch.tensor(
            [f["numeric_features"] for f in features],
            dtype=torch.float32,
        )
        stripped = [{k: v for k, v in f.items() if k != "numeric_features"} for f in features]
        batch = self.base_collator(stripped)
        batch["numeric_features"] = numeric_features
        return batch


class UnfreezeBaseModelCallback(TrainerCallback):
    def __init__(self, freeze_base_epochs):
        self.freeze_base_epochs = int(freeze_base_epochs)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is None or self.freeze_base_epochs <= 0:
            return control

        if (
            getattr(model, "is_base_frozen", False)
            and state.epoch is not None
            and state.epoch >= self.freeze_base_epochs
        ):
            model.unfreeze_base_model()
            logging.info(
                "Unfroze pretrained encoder at epoch %.2f (freeze_base_epochs=%d).",
                state.epoch,
                self.freeze_base_epochs,
            )
        return control


class NumericTrainer(Trainer):
    def __init__(self, *args, encoder_learning_rate, ffn_learning_rate, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_learning_rate = float(encoder_learning_rate)
        self.ffn_learning_rate = float(ffn_learning_rate)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        encoder_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        optimizer_grouped_parameters = []
        if encoder_params:
            optimizer_grouped_parameters.append(
                {
                    "params": encoder_params,
                    "lr": self.encoder_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
        if head_params:
            optimizer_grouped_parameters.append(
                {
                    "params": head_params,
                    "lr": self.ffn_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def _parse_feature_list(raw_value, field_name, model_name):
    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(
            f"Model '{model_name}' is missing '{field_name}' in model parameters CSV."
        )
    return [part.strip() for part in str(raw_value).split(";") if part.strip()]


def _parse_optional_float(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text == "":
        return None
    return float(text)


def _resolve_learning_rates(row):
    # Backward compatibility: if ffn LR is absent, reuse legacy learning_rate.
    fallback_lr = _parse_optional_float(row.get("learning_rate"))
    ffn_lr = _parse_optional_float(row.get("ffn_learning_rate"))
    if ffn_lr is None:
        if fallback_lr is None:
            raise ValueError("Missing both 'ffn_learning_rate' and fallback 'learning_rate'.")
        ffn_lr = fallback_lr

    # If encoder LR is absent, use FFN LR (single-LR joint training).
    encoder_lr = _parse_optional_float(row.get("encoder_learning_rate"))
    if encoder_lr is None:
        encoder_lr = ffn_lr

    return ffn_lr, encoder_lr


def _to_float(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _add_numeric_features(batch, numeric_feature_cols):
    rows = zip(*(batch[col] for col in numeric_feature_cols))
    return {
        "numeric_features": [
            [_to_float(v) for v in row]
            for row in rows
        ]
    }


def _preprocess_dataset_with_numeric(ds_dict, text_cols, numeric_feature_cols, sep_token):
    first_split = next(iter(ds_dict.values()))
    all_columns = first_split.column_names
    cols_to_keep = {"input_text", "label", "numeric_features"}
    cols_to_remove = [c for c in all_columns if c not in cols_to_keep and c != "GLMM_score"]

    ds_dict = ds_dict.map(
        merge_cols,
        batched=True,
        fn_kwargs={"cols_to_merge": text_cols, "sep_token": sep_token},
        desc="Formatting input text",
    )
    ds_dict = ds_dict.map(
        _add_numeric_features,
        batched=True,
        fn_kwargs={"numeric_feature_cols": numeric_feature_cols},
        desc="Collecting numeric features",
    )
    ds_dict = ds_dict.rename_column("GLMM_score", "label").remove_columns(cols_to_remove)
    return ds_dict


def _compute_train_normalization_stats(train_numeric):
    train_arr = np.asarray(train_numeric, dtype=np.float32)
    means = np.nanmean(train_arr, axis=0)
    vars_ = np.nanvar(train_arr, axis=0)
    means = np.nan_to_num(means, nan=0.0)
    vars_ = np.nan_to_num(vars_, nan=1.0)
    vars_ = np.maximum(vars_, 1e-8)
    return means, vars_


def _normalize_numeric_features(ds_dict, means, vars_):
    means = np.asarray(means, dtype=np.float32)
    std = np.sqrt(np.asarray(vars_, dtype=np.float32))

    def normalize_batch(batch):
        arr = np.asarray(batch["numeric_features"], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=means)
        normalized = (arr - means) / std
        return {"numeric_features": normalized.tolist()}

    return ds_dict.map(
        normalize_batch,
        batched=True,
        desc="Normalizing numeric features",
    )


def run_finetune_num(model_params_path, models_to_run, seed):
    """
    Fine-tune a text+numeric regression model using pretrained text embeddings
    concatenated with normalized numeric features.
    """
    set_seed(seed)

    for row in load_model_params(model_params_path, models_to_run):
        model_name = row["model_name"]
        l1 = row["L1"]

        try:
            logging.info("Fine-tuning numeric model: %s...", model_name)

            data_files = load_data_paths(DATA_DIR, l1, "finetune")
            hf_dataset = load_dataset("csv", data_files=data_files)

            tokenizer = AutoTokenizer.from_pretrained(row["pretrained_model"], use_fast=True)
            text_cols = _parse_feature_list(row.get("component_order"), "component_order", model_name)
            numeric_feature_cols = _parse_feature_list(
                row.get("numeric_features"), "numeric_features", model_name
            )
            sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "

            preprocessed_ds = _preprocess_dataset_with_numeric(
                hf_dataset,
                text_cols,
                numeric_feature_cols,
                sep_token,
            )

            means, vars_ = _compute_train_normalization_stats(
                preprocessed_ds["train"]["numeric_features"]
            )
            normalized_ds = _normalize_numeric_features(preprocessed_ds, means, vars_)

            tokenized_ds = normalized_ds.map(
                lambda x: tokenizer(x["input_text"], truncation=True),
                batched=True,
                desc="Tokenizing input text",
            )

            freeze_base_epochs = int(row.get("freeze_base_epochs", 0) or 0)
            ffn_learning_rate, encoder_learning_rate = _resolve_learning_rates(row)
            model_config = AutoConfig.from_pretrained(row["pretrained_model"])
            model_config.pretrained_model_name_or_path = row["pretrained_model"]
            model_config.num_numeric_features = len(numeric_feature_cols)
            model_config.numeric_feature_names = numeric_feature_cols
            model_config.numeric_feature_means = means.tolist()
            model_config.numeric_feature_vars = vars_.tolist()
            model_config.freeze_base_epochs = freeze_base_epochs

            training_args = TrainingArguments(
                output_dir=str(MODELS_DIR / model_name),
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                num_train_epochs=int(row["epochs"]),
                per_device_train_batch_size=int(row["batch_size"]),
                per_device_eval_batch_size=int(row["batch_size"]),
                learning_rate=ffn_learning_rate,
                weight_decay=float(row["weight_decay"]),
                warmup_ratio=float(row["warmup_ratio"]),
                load_best_model_at_end=True,
                report_to="none",
                seed=seed,
            )

            logging.info(
                "Using learning rates for %s: ffn=%.8f, encoder=%.8f",
                model_name,
                ffn_learning_rate,
                encoder_learning_rate,
            )

            trainer = NumericTrainer(
                model_init=lambda: NumericRegressionModel(model_config),
                args=training_args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["validation"],
                data_collator=NumericDataCollator(tokenizer),
                compute_metrics=compute_metrics,
                callbacks=[UnfreezeBaseModelCallback(freeze_base_epochs)],
                ffn_learning_rate=ffn_learning_rate,
                encoder_learning_rate=encoder_learning_rate,
            )

            trainer.train()
            trainer.save_model(MODELS_DIR / model_name)
            logging.info(
                "Numeric model %s fine-tuned and saved at %s",
                model_name,
                MODELS_DIR / model_name,
            )
            cleanup_trainer_memory(trainer, tokenized_ds, normalized_ds, preprocessed_ds)

        except Exception:
            logging.exception("Failed numeric model %s", model_name)
            raise
