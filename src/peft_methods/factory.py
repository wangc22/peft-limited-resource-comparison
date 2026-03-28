from peft import LoraConfig, TaskType, get_peft_model


def apply_peft_method(model, config: dict):
    method = config["peft"]["method"].lower()

    if method == "lora":
        lora_cfg = config["peft"]["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        model = get_peft_model(model, peft_config)
        return model

    raise NotImplementedError(
        f"PEFT method '{method}' is not implemented yet. Start with 'lora'."
    )


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable / total
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_percent": ratio,
    }
