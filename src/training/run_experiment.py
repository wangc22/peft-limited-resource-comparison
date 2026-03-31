import time
import torch
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json
from src.datasets.classification import load_sst2_dataset
from src.models.classifier import load_tokenizer, load_sequence_classifier
from src.peft_methods.factory import apply_peft_method, count_trainable_parameters
from src.evaluation.metrics import compute_metrics


def run_experiment(config_path: str):
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])

    output_root = config["experiment"]["output_dir"]
    exp_name = config["experiment"]["name"]

    ensure_dir(f"{output_root}/logs")
    ensure_dir(f"{output_root}/metrics")
    ensure_dir(f"{output_root}/checkpoints")

    dataset = load_sst2_dataset()
    model_name = config["model"]["backbone"]
    tokenizer = load_tokenizer(model_name)

    text_key = config["task"]["text_key"]
    label_key = config["task"]["label_key"]
    max_length = config["task"]["max_length"]

    def preprocess(examples):
        return tokenizer(
            examples[text_key],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column(label_key, "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = load_sequence_classifier(
        model_name=model_name,
        num_labels=config["model"]["num_labels"]
    )

    model = apply_peft_method(model, config)

    param_stats = count_trainable_parameters(model)
    print("Parameter stats:", param_stats)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"{output_root}/checkpoints/{exp_name}",
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(config["training"]["batch_size"]),
        num_train_epochs=int(config["training"]["num_train_epochs"]),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        eval_strategy=config["training"]["evaluation_strategy"],
        save_strategy=config["training"]["save_strategy"],
        logging_steps=int(config["training"]["logging_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    skip_training = config.get("baseline", {}).get("skip_training", False)

    if skip_training:
        peak_memory_before = 0
        peak_memory_after = 0
        total_train_time = 0.0
        train_result = type("obj", (), {"training_loss": None})()
    else:
        start_time = time.time()
        peak_memory_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        train_result = trainer.train()
        peak_memory_after = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        total_train_time = time.time() - start_time

    eval_metrics = trainer.evaluate(tokenized["validation"])

    final_metrics = {
        "experiment_name": exp_name,
        "method": config["peft"]["method"],
        "dataset": config["task"]["dataset"],
        "seed": config["experiment"]["seed"],
        **param_stats,
        "train_runtime_seconds": total_train_time,
        "peak_memory_bytes": max(peak_memory_before, peak_memory_after),
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_metrics,
    }

    save_json(final_metrics, f"{output_root}/metrics/{exp_name}.json")
    save_json(config, f"{output_root}/logs/{exp_name}_config.json")

    print("Run complete.")
    print(final_metrics)