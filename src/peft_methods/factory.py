from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    TaskType,
    get_peft_model,
)


def apply_peft_method(model, config: dict):
    method = config["peft"]["method"].lower()

    if method == "none":
        #return the model without any PEFT method
        return model

    if method == "full":
        #train all parameters
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

    if method == "prefix":
        prefix_cfg = config["peft"]["prefix"]
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=prefix_cfg["num_virtual_tokens"],
        )
        return get_peft_model(model, peft_config)

    if method == "ptuning":
        ptuning_cfg = config["peft"]["ptuning"]
        peft_config = PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=ptuning_cfg["num_virtual_tokens"],
            encoder_hidden_size=ptuning_cfg["encoder_hidden_size"],
        )
        return get_peft_model(model, peft_config)

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