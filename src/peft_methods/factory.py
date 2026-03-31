from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)


def apply_peft_method(model, config: dict, tokenizer=None):
    method = config["peft"]["method"].lower()

    if method == "none":
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        return model

    if method == "full":
        for param in model.parameters():
            param.requires_grad = True
        return model

    if method == "lora":
        lora_cfg = config["peft"]["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
        )
        return get_peft_model(model, peft_config)

    if method == "prefix_tuning":
        prefix_cfg = config["peft"]["prefix_tuning"]
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=prefix_cfg["num_virtual_tokens"],
        )
        return get_peft_model(model, peft_config)

    if method == "prompt_tuning":
        prompt_cfg = config["peft"]["prompt_tuning"]
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=prompt_cfg["num_virtual_tokens"],
        )
        return get_peft_model(model, peft_config)

    if method == "adapter":
        raise NotImplementedError("Adapter is not implemented yet.")

    raise NotImplementedError(f"PEFT method '{method}' is not implemented yet.")


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100 * trainable / total
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_percent": ratio,
    }