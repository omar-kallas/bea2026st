import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from finetune_num import NumericDataCollator, NumericRegressionModel
from utils import (
    cleanup_trainer_memory,
    load_data_paths,
    load_model_params,
    merge_cols,
    save_predictions,
)

# Fixed paths
DATA_DIR = Path("data_enriched/")
MODELS_DIR = Path("models/")
PRED_DIR = Path("predictions/")


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
    return {"numeric_features": [[_to_float(v) for v in row] for row in rows]}


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


def _normalize_numeric_features(ds_dict, means, vars_):
    means = np.asarray(means, dtype=np.float32)
    std = np.sqrt(np.maximum(np.asarray(vars_, dtype=np.float32), 1e-8))

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


def run_predict_num(model_params_path, models_to_run, dataset_split):
    """
    Run predictions for models trained with finetune_num.py and save as CSV files.
    """
    kvl_splits_to_hf = {
        "dev": "validation",
        "test": "test",
    }
    splits_to_run = ["dev", "test"] if dataset_split == "both" else [dataset_split]

    for row in load_model_params(model_params_path, models_to_run):
        model_name = row["model_name"]
        track = row["track"]
        l1 = row["L1"]
        l1_datasets = ["es", "de", "cn"] if l1 == "xx" else [l1]

        model_path = MODELS_DIR / model_name
        model = NumericRegressionModel.from_pretrained(model_path)
        model.eval()

        numeric_feature_cols = list(getattr(model.config, "numeric_feature_names", []))
        numeric_means = list(getattr(model.config, "numeric_feature_means", []))
        numeric_vars = list(getattr(model.config, "numeric_feature_vars", []))
        if not numeric_feature_cols:
            raise ValueError(
                f"Model '{model_name}' is missing numeric_feature_names in saved config."
            )
        if len(numeric_feature_cols) != len(numeric_means) or len(numeric_feature_cols) != len(numeric_vars):
            raise ValueError(
                f"Model '{model_name}' has inconsistent numeric normalization metadata in config."
            )

        tokenizer = AutoTokenizer.from_pretrained(row["pretrained_model"], use_fast=True)
        text_cols = [part.strip() for part in str(row["component_order"]).split(";") if part.strip()]
        sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "

        for l1 in l1_datasets:
            try:
                logging.info("Predicting with numeric model %s on L1=%s", model_name, l1)
                data_files = load_data_paths(DATA_DIR, l1, "predict", dataset_split=dataset_split)

                if not data_files:
                    logging.warning(
                        "No data files found for %s %s set(s). Skipping this L1.",
                        l1,
                        dataset_split,
                    )
                    continue

                hf_dataset = load_dataset("csv", data_files=data_files)
                preprocessed_ds = _preprocess_dataset_with_numeric(
                    hf_dataset,
                    text_cols,
                    numeric_feature_cols,
                    sep_token,
                )
                normalized_ds = _normalize_numeric_features(
                    preprocessed_ds,
                    numeric_means,
                    numeric_vars,
                )
                tokenized_ds = normalized_ds.map(
                    lambda x: tokenizer(x["input_text"], truncation=True),
                    batched=True,
                    desc="Tokenizing input text",
                )

                trainer = Trainer(
                    model=model,
                    data_collator=NumericDataCollator(tokenizer),
                    args=TrainingArguments(
                        output_dir=str(MODELS_DIR / model_name),
                        report_to="none",
                    ),
                )

                for split_name in splits_to_run:
                    logging.info("Getting %s predictions...", split_name)
                    hf_key = kvl_splits_to_hf.get(split_name, split_name)
                    if hf_key not in tokenized_ds:
                        logging.warning("No data found for %s %s set. Skipping.", l1, split_name)
                        continue

                    ds = tokenized_ds[hf_key]
                    with torch.no_grad():
                        preds = trainer.predict(ds).predictions.flatten()

                    item_ids = hf_dataset[hf_key]["item_id"]
                    save_dir = PRED_DIR / track / split_name / l1
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{model_name}_preds.csv"
                    save_predictions(save_path, item_ids, preds)
                    logging.info(
                        "Saved predictions for %s on %s %s set: %s",
                        model_name,
                        l1,
                        split_name,
                        save_path,
                    )

                cleanup_trainer_memory(trainer, tokenized_ds, normalized_ds, preprocessed_ds)

            except Exception:
                logging.exception(
                    "Numeric predictions failed for %s on %s %s set(s)",
                    model_name,
                    l1,
                    dataset_split,
                )
                raise
