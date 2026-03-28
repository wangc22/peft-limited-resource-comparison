from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def load_sequence_classifier(model_name: str, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model
